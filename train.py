"""Train and evaluate the award models.

Primary task: All-NBA selection (binary), ranked within season (k = that season's number of slots:
10 for 1980-1988, 15 from 1989). Secondary: All-NBA vote share (regression), All-Star selection,
MVP vote share.

Protocol:
- Train on 1980-2019. Tune with forward-chaining CV inside that range (blocks in config.CV_BLOCKS,
  each block is scored by a model trained on all seasons before it).
- The model to serialise is chosen by CV PR-AUC, never by the holdout.
- The 2020+ labelled seasons are scored once at the end. Baselines are scored on the same rows.
"""
from __future__ import annotations

import itertools
import json
import warnings

import joblib
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.ensemble import (HistGradientBoostingClassifier, HistGradientBoostingRegressor,
                              RandomForestClassifier)
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, mean_absolute_error
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

import config as C
import evaluate as E

warnings.filterwarnings("ignore", category=UserWarning)

GRIDS = {
    "logreg": [{"C": c} for c in (0.01, 0.1, 1.0, 10.0)],
    "random_forest": [{"max_depth": d, "min_samples_leaf": m} for d, m in itertools.product((None, 8), (1, 5, 20))],
    "hist_gb": [{"learning_rate": lr, "max_iter": it, "max_leaf_nodes": n, "min_samples_leaf": m, "l2_regularization": l2}
                for (lr, it), n, m, l2 in itertools.product(((0.03, 400), (0.1, 150)), (8, 31), (20, 80), (0.0, 1.0))],
}
BASELINES = {"rank_by_vorp": "vorp", "rank_by_ws": "ws", "rank_by_pts": "pts_per_game"}


def make_model(name: str, params: dict):
    if name == "logreg":
        return make_pipeline(SimpleImputer(strategy="median"), StandardScaler(),
                             LogisticRegression(class_weight="balanced", max_iter=5000, **params))
    if name == "random_forest":
        return make_pipeline(SimpleImputer(strategy="median"),
                             RandomForestClassifier(n_estimators=400, max_features="sqrt", n_jobs=-1,
                                                    random_state=C.RANDOM_STATE, **params))
    if name == "hist_gb":
        return HistGradientBoostingClassifier(early_stopping=False, random_state=C.RANDOM_STATE, **params)
    if name == "hist_gb_reg":
        return HistGradientBoostingRegressor(early_stopping=False, random_state=C.RANDOM_STATE, **params)
    raise ValueError(name)


def k_map_from_labels(table: pd.DataFrame, label_col: str, label_counts: pd.Series | None = None) -> dict:
    """Slots per season = number of real selections (incl. any dropped by the population filter)."""
    if label_counts is not None:
        return label_counts.to_dict()
    return table.groupby("season")[label_col].sum().astype(int).to_dict()


def split(table, label_col, last_labelled):
    lab = table[table[label_col].notna()]
    train = lab[lab["season"].between(C.FIRST_SEASON, C.TRAIN_LAST)]
    test = lab[lab["season"].between(C.TEST_FIRST, last_labelled)]
    assert train["season"].max() < test["season"].min(), "temporal leak: train overlaps test"
    return train, test


def forward_cv(train, label_col, name, params, k_map, features=C.FEATURES):
    rows, oof = [], []
    for lo, hi in C.CV_BLOCKS:
        tr = train[train["season"] < lo]
        va = train[train["season"].between(lo, hi)]
        m = make_model(name, params).fit(tr[features], tr[label_col].astype(int))
        p = pd.Series(m.predict_proba(va[features])[:, 1], index=va.index)
        oof.append(p)
        met = E.classification_metrics(va, p, label_col, k_map)
        rows.append({"block": f"{lo}-{hi}", **met})
    df = pd.DataFrame(rows)
    return {"cv_pr_auc": df["pr_auc"].mean(), "cv_precision_at_k": df["precision_at_k"].mean(),
            "cv_roc_auc": df["roc_auc"].mean(), "cv_brier": df["brier"].mean(), "folds": rows,
            "oof": pd.concat(oof)}


def tune(train, label_col, k_map, log=print, grids=GRIDS, features=C.FEATURES):
    """Returns (all CV results, best params per model family, out-of-fold predictions of each best config)."""
    results, best, oof = [], {}, {}
    for name, grid in grids.items():
        fam = []
        for params in grid:
            r = forward_cv(train, label_col, name, params, k_map, features)
            fam.append((params, r))
            results.append({"model": name, "params": json.dumps(params),
                            **{k: v for k, v in r.items() if k not in ("folds", "oof")}})
            log(f"[train] cv {name:14s} {json.dumps(params):110s} PR-AUC {r['cv_pr_auc']:.3f}  P@k {r['cv_precision_at_k']:.3f}")
        params, r = max(fam, key=lambda x: (x[1]["cv_pr_auc"], x[1]["cv_precision_at_k"]))
        best[name] = {"params": params, "cv_pr_auc": r["cv_pr_auc"], "cv_precision_at_k": r["cv_precision_at_k"],
                      "cv_roc_auc": r["cv_roc_auc"], "cv_folds": r["folds"]}
        oof[name] = r["oof"]
    return pd.DataFrame(results), best, oof


def fit_calibrated(name, params, train, label_col, oof, features=C.FEATURES):
    base = make_model(name, params).fit(train[features], train[label_col].astype(int))
    a, b = E.PlattCalibrated.fit_params(oof, train.loc[oof.index, label_col].astype(int))
    return E.PlattCalibrated(base, a, b)


def baseline_cv(train, label_col, k_map):
    out = {}
    for bname, col in BASELINES.items():
        aps, pks = [], []
        for lo, hi in C.CV_BLOCKS:
            va = train[train["season"].between(lo, hi)]
            met = E.classification_metrics(va, va[col].astype(float), label_col, k_map, prob=False)
            aps.append(met["pr_auc"]); pks.append(met["precision_at_k"])
        out[bname] = {"cv_pr_auc": float(np.mean(aps)), "cv_precision_at_k": float(np.mean(pks))}
    return out


def score_holdout(test, scores: dict, label_col, k_map, prob_names=()):
    """scores: {name: Series}. Returns overall rows (raw / with 65-game mask / positional) and per-season hits."""
    overall, per_season = [], []
    for name, s in scores.items():
        for variant, s_v, positional in (("raw", s, False), ("eligibility_mask", E.apply_eligibility(test, s), False),
                                         ("mask+positional", E.apply_eligibility(test, s), True)):
            met = E.classification_metrics(test, s_v, label_col, k_map, prob=name in prob_names, positional=positional)
            ps = E.per_season_hits(test, s_v, label_col, k_map, positional)
            early, late = ps[ps["season"] < C.POSITIONLESS_FROM], ps[ps["season"] >= C.POSITIONLESS_FROM]
            met["p@k_2020_2023"] = float(early["hits"].sum() / early["k"].sum()) if len(early) else np.nan
            met["p@k_2024plus"] = float(late["hits"].sum() / late["k"].sum()) if len(late) else np.nan
            overall.append({"scorer": name, "variant": variant, **met})
            per_season.append(ps.assign(scorer=name, variant=variant))
    return pd.DataFrame(overall), pd.concat(per_season, ignore_index=True)


# --------------------------------------------------------------------------- secondary tasks
def train_vote_share(table, last_labelled, k_map, hgb_params, log=print):
    lab = table[table["all_nba_share"].notna()]
    tr = lab[lab["season"].between(C.FIRST_SEASON, C.TRAIN_LAST)]
    te = lab[lab["season"].between(C.TEST_FIRST, last_labelled)]
    params = {k: v for k, v in hgb_params.items()}
    m = make_model("hist_gb_reg", params).fit(tr[C.FEATURES], tr["all_nba_share"])
    pred = pd.Series(m.predict(te[C.FEATURES]), index=te.index)
    masked = E.apply_eligibility(te, pred)
    voted = te[te["all_nba_share"] > 0]
    res = {
        "mae_all_rows": float(mean_absolute_error(te["all_nba_share"], pred)),
        "mae_vote_getters": float(mean_absolute_error(voted["all_nba_share"], pred.loc[voted.index])),
        "spearman_within_season_vote_getters": float(np.mean([
            spearmanr(g["all_nba_share"], pred.loc[g.index]).statistic for _, g in voted.groupby("season")])),
        "selection_precision_at_k_masked": E.classification_metrics(te, masked, "made_all_nba", k_map, prob=False)["precision_at_k"],
        "team_assignment_accuracy_masked": E.team_assignment_accuracy(te, masked, k_map),
    }
    log(f"[train] vote-share regressor: {res}")
    return m, res, pred


def run_all_star(table, k_map, log=print):
    """All-Star selection (k = real selections that season, incl. injury replacements). 1999 has no game."""
    lab = table[table["all_star"].notna()]
    last = int(lab["season"].max())
    tr = lab[lab["season"].between(C.FIRST_SEASON, C.TRAIN_LAST)]
    te = lab[lab["season"].between(C.TEST_FIRST, last)]
    grids = {"logreg": [{"C": 0.1}, {"C": 1.0}], "hist_gb": GRIDS["hist_gb"][::2]}
    _, best, oof = tune(tr, "all_star", k_map, log=lambda *_: None, grids=grids)
    scores, models = {}, {}
    for name, b in best.items():
        m = fit_calibrated(name, b["params"], tr, "all_star", oof[name])
        models[name] = m
        scores[name] = pd.Series(m.predict_proba(te[C.FEATURES])[:, 1], index=te.index)
    scores["rank_by_pts"] = te["pts_per_game"].astype(float)
    scores["rank_by_vorp"] = te["vorp"].astype(float)
    rows = []
    for name, s in scores.items():
        met = E.classification_metrics(te, s, "all_star", k_map, prob=name in best)
        rows.append({"scorer": name, **met})
    res = pd.DataFrame(rows)
    chosen = max(best, key=lambda n: best[n]["cv_pr_auc"])
    log(f"[train] All-Star (test {C.TEST_FIRST}-{last}, k = real selections incl. injury replacements):\n"
        + res.to_string(index=False))
    return models[chosen], chosen, best, res, last


def run_mvp(table, hgb_params, log=print):
    lab = table[table["mvp_share"].notna()]
    last = int(lab["season"].max())
    tr = lab[lab["season"].between(C.FIRST_SEASON, C.TRAIN_LAST)]
    te = lab[lab["season"].between(C.TEST_FIRST, last)]
    m = make_model("hist_gb_reg", hgb_params).fit(tr[C.FEATURES], tr["mvp_share"])
    pred = pd.Series(m.predict(te[C.FEATURES]), index=te.index)
    pred_m = E.apply_eligibility(te, pred)
    rows = []
    for name, s in {"hist_gb_reg (masked)": pred_m, "hist_gb_reg (raw)": pred,
                    "rank_by_vorp": te["vorp"].astype(float), "rank_by_ws": te["ws"].astype(float),
                    "rank_by_per": te["per"].astype(float), "rank_by_bpm": te["bpm"].astype(float)}.items():
        per = []
        for season, g in te.assign(_s=s).groupby("season"):
            top = g.sort_values("_s", ascending=False)
            winner = g[g["mvp_winner"] == 1]
            per.append({"season": season, "pred_winner": top.iloc[0]["player"],
                        "real_winner": winner["player"].iloc[0] if len(winner) else None,
                        "hit": bool(top.iloc[0]["mvp_winner"] == 1),
                        "real_winner_pred_rank": int(top.index.get_loc(winner.index[0]) + 1) if len(winner) else None,
                        "top5_overlap": int((top.head(5)["mvp_rank"] <= 5).sum())})
        per = pd.DataFrame(per)
        rows.append({"scorer": name, "winner_hits": int(per["hit"].sum()), "seasons": len(per),
                     "mean_top5_overlap": float(per["top5_overlap"].mean()),
                     "per_season": per.to_dict("records")})
    log("[train] MVP (test {}-{}):\n".format(C.TEST_FIRST, last)
        + pd.DataFrame([{k: v for k, v in r.items() if k != "per_season"} for r in rows]).to_string(index=False))
    return m, rows, last


# --------------------------------------------------------------------------- main
def main(table: pd.DataFrame | None = None, label_counts: dict | None = None, log=print, out_models=C.MODELS_DIR,
         out_reports=C.REPORTS_DIR):
    if table is None:
        table = pd.read_csv(C.TABLE_PATH)
    out_models.mkdir(parents=True, exist_ok=True)
    out_reports.mkdir(parents=True, exist_ok=True)
    label = C.PRIMARY_TASK
    lab_seasons = table.loc[table[label].notna(), "season"]
    last_labelled = int(lab_seasons.max())
    k_all_nba = label_counts["all_nba"] if label_counts else k_map_from_labels(table, label)
    train, test = split(table, label, last_labelled)
    log(f"[train] All-NBA: train {train['season'].min()}-{train['season'].max()} ({len(train)} rows, "
        f"{int(train[label].sum())} positives); test {test['season'].min()}-{test['season'].max()} "
        f"({len(test)} rows, {int(test[label].sum())} positives); features {len(C.FEATURES)}")

    cv_df, best, oof = tune(train, label, k_all_nba, log)
    base_cv = baseline_cv(train, label, k_all_nba)
    for b, r in base_cv.items():
        log(f"[train] cv {b:14s} PR-AUC {r['cv_pr_auc']:.3f}  P@k {r['cv_precision_at_k']:.3f}")
    cv_df.to_csv(out_reports / "cv_results.csv", index=False)
    chosen = max(best, key=lambda n: (best[n]["cv_pr_auc"], best[n]["cv_precision_at_k"]))
    log(f"[train] chosen by CV PR-AUC: {chosen} {best[chosen]['params']}")

    # ---- ablation: drop voter-history features (CV only + holdout, reported, not used for selection)
    no_hist = [f for f in C.FEATURES if not f.startswith(("prior_", "prev_"))]
    abl = forward_cv(train, label, chosen, best[chosen]["params"], k_all_nba, no_hist)

    # ---- final fits on all training seasons, single holdout evaluation
    models, scores, brier_uncal = {}, {}, {}
    for name, b in best.items():
        m = fit_calibrated(name, b["params"], train, label, oof[name])
        models[name] = m
        scores[name] = pd.Series(m.predict_proba(test[C.FEATURES])[:, 1], index=test.index)
        p_raw = m.base.predict_proba(test[C.FEATURES])[:, 1]
        brier_uncal[name] = float(((p_raw - test[label]) ** 2).mean())
    m_abl = fit_calibrated(chosen, best[chosen]["params"], train, label, abl["oof"], no_hist)
    scores[f"{chosen}_no_voter_history"] = pd.Series(m_abl.predict_proba(test[no_hist])[:, 1], index=test.index)
    for bname, col in BASELINES.items():
        scores[bname] = test[col].astype(float)
    prob_names = set(best) | {f"{chosen}_no_voter_history"}
    overall, per_season = score_holdout(test, scores, label, k_all_nba, prob_names)
    overall.to_csv(out_reports / "test_overall.csv", index=False)
    per_season.to_csv(out_reports / "test_per_season.csv", index=False)
    log("[train] holdout (All-NBA):\n" + overall[overall["variant"] != "mask+positional"].round(3).to_string(index=False))

    p_best = scores[chosen]
    rel = E.reliability_table(test[label], p_best)
    rel.to_csv(out_reports / "calibration.csv", index=False)
    team_acc = E.team_assignment_accuracy(test, E.apply_eligibility(test, p_best), k_all_nba)
    prob_sum = test.assign(p=p_best).groupby("season")["p"].sum().round(2).to_dict()

    preds = test[["season", "player", "player_id", "team", "pos", "g", "mp", "vorp", "bpm", "ws", "team_win_pct",
                  "prior_all_nba", "made_all_nba", "all_nba_team", "all_nba_share"]].copy()
    for name in best:
        preds[f"p_{name}"] = scores[name]
    preds["picked_best_masked"] = E.top_k(test, E.apply_eligibility(test, p_best), k_all_nba)
    preds["rank_best_masked"] = E.apply_eligibility(test, p_best).groupby(test["season"]).rank(ascending=False, method="min")
    preds.sort_values(["season", "rank_best_masked"]).to_csv(out_reports / "test_predictions.csv", index=False)

    # ---- secondary tasks
    hgb_params = dict(best["hist_gb"]["params"])
    vs_model, vs_res, vs_pred = train_vote_share(table, last_labelled, k_all_nba, hgb_params, log)
    as_model, as_name, as_best, as_res, as_last = run_all_star(
        table, label_counts["all_star"] if label_counts else k_map_from_labels(table, "all_star"), log)
    mvp_model, mvp_rows, mvp_last = run_mvp(table, hgb_params, log)

    # ---- serialise
    meta = {"features": C.FEATURES, "train_seasons": [C.FIRST_SEASON, C.TRAIN_LAST],
            "eligibility_rule": {"from_season": C.GAMES_RULE_FROM, "min_g": C.GAMES_RULE_MIN_G},
            "population_filter": {"min_g": C.MIN_G, "min_mp": C.MIN_MP, "scaled_by_season_length": True}}
    joblib.dump({"model": models[chosen], "name": chosen, "params": best[chosen]["params"], **meta},
                out_models / "all_nba_best.joblib")
    for name, m in models.items():
        joblib.dump({"model": m, "name": name, "params": best[name]["params"], **meta}, out_models / f"all_nba_{name}.joblib")
    joblib.dump({"model": vs_model, "name": "hist_gb_reg", **meta}, out_models / "all_nba_vote_share.joblib")
    joblib.dump({"model": as_model, "name": as_name, **meta}, out_models / "all_star.joblib")
    joblib.dump({"model": mvp_model, "name": "hist_gb_reg", **meta}, out_models / "mvp_share.joblib")

    metrics = {
        "all_nba": {"train": [int(train.season.min()), int(train.season.max())],
                    "test": [int(test.season.min()), int(test.season.max())],
                    "chosen_model": chosen, "best_by_model": best, "baselines_cv": base_cv,
                    "brier_uncalibrated_holdout": brier_uncal,
                    "calibration": {n: {"a": m.a, "b": m.b} for n, m in models.items()},
                    "ablation_no_voter_history_cv": {k: v for k, v in abl.items() if k not in ("folds", "oof")},
                    "holdout": overall.to_dict("records"), "team_assignment_accuracy_best_masked": team_acc,
                    "sum_of_probabilities_per_test_season": prob_sum},
        "all_nba_vote_share": vs_res,
        "all_star": {"chosen_model": as_name, "test_last": as_last,
                     "best_by_model": as_best, "holdout": as_res.to_dict("records")},
        "mvp": {"test_last": mvp_last, "holdout": mvp_rows},
    }
    (out_reports / "metrics.json").write_text(json.dumps(metrics, indent=2, default=str))
    log(f"[train] wrote models to {out_models} and metrics to {out_reports / 'metrics.json'}")
    return metrics


if __name__ == "__main__":
    import features as F
    raw = F.load_raw()
    L = F.label_tables(raw)
    counts = {"all_nba": L["all_nba"].groupby("season").size().to_dict(),
              "all_star": L["all_star"].groupby("season").size().to_dict()}
    main(label_counts=counts)
