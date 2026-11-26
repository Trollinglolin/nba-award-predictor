"""v2: do the new feature groups help? Feature-set comparison for All-NBA and MVP, then the v2 models.

Feature sets (C.FEATURE_GROUPS / C.NBA_GROUPS):
  A-sets train from 1980 (Kaggle groups that exist for every season),
  B-sets train from 1997 (adds B-Ref on/off and, if fetched, the nba_api groups).
Every set is tuned and scored on the SAME forward-chaining CV blocks (C.V2_CV_BLOCKS, 2005-2019), so CV numbers are
comparable across sets. The v2 model is chosen by CV PR-AUC only (with a parsimony rule: among sets within 0.002 of the
best, the one with the fewest features). The 2020+ holdout is reported for every set but never used to choose.
"""
from __future__ import annotations

import json
import warnings

import joblib
import numpy as np
import pandas as pd

import config as C
import evaluate as E
import train as T

warnings.filterwarnings("ignore")

LR_GRID = [{"C": c} for c in (0.03, 0.1, 0.3)]
HGB_GRID = [{"learning_rate": 0.1, "max_iter": 150, "max_leaf_nodes": 8, "min_samples_leaf": 20, "l2_regularization": 1.0},
            {"learning_rate": 0.03, "max_iter": 400, "max_leaf_nodes": 8, "min_samples_leaf": 50, "l2_regularization": 1.0},
            {"learning_rate": 0.1, "max_iter": 150, "max_leaf_nodes": 15, "min_samples_leaf": 50, "l2_regularization": 1.0}]
PARSIMONY = 0.002


def group_coverage(table, cols, first, last=C.TRAIN_LAST) -> float:
    tr = table[table["season"].between(first, last)]
    return float(tr[cols].notna().all(axis=1).mean()) if len(tr) else 0.0


def feature_sets(table, log=print) -> list[dict]:
    G, base = C.FEATURE_GROUPS, list(C.BASE_FEATURES)
    kag = G["fatigue"] + G["jump"] + G["teammates"] + G["narrative"]
    sets = [{"name": "A base (v1 features)", "features": base, "first": C.FIRST_SEASON}]
    for g in ["fatigue", "jump", "teammates", "narrative"]:
        sets.append({"name": f"A +{g}", "features": base + G[g], "first": C.FIRST_SEASON})
    sets.append({"name": "A +all Kaggle groups", "features": base + kag, "first": C.FIRST_SEASON})
    sets += [{"name": "B base (v1 features)", "features": base, "first": C.V2_TRAIN_FIRST},
             {"name": "B +all Kaggle groups", "features": base + kag, "first": C.V2_TRAIN_FIRST},
             {"name": "B +all Kaggle +onoff", "features": base + kag + G["onoff"], "first": C.V2_TRAIN_FIRST}]
    usable = []
    for g, cols in C.NBA_GROUPS.items():
        cov = group_coverage(table, cols, C.V2_TRAIN_FIRST)
        log(f"[v2] nba_api group {g:13s} coverage of 1997-{C.TRAIN_LAST} training rows: {cov:.0%}")
        if cov >= C.MIN_GROUP_COVERAGE:
            usable.append(g)
            sets.append({"name": f"B +all Kaggle +onoff +{g}", "features": base + kag + G["onoff"] + cols,
                         "first": C.V2_TRAIN_FIRST})
    if usable:
        allnba = sum((C.NBA_GROUPS[g] for g in usable), [])
        sets.append({"name": "B +everything", "features": base + kag + G["onoff"] + allnba, "first": C.V2_TRAIN_FIRST})
    return sets


def _with_blocks(fn):
    saved = C.CV_BLOCKS
    C.CV_BLOCKS = C.V2_CV_BLOCKS
    try:
        return fn()
    finally:
        C.CV_BLOCKS = saved


def paired_bootstrap(a: pd.Series, b: pd.Series, n_tests: int = 1, n=4000, seed=C.RANDOM_STATE):
    """Mean of (a - b) over seasons and its bootstrap CI (resampling seasons), Bonferroni-adjusted for n_tests
    comparisons: the CI is (1 - 0.05 / n_tests). A feature set is adopted only if the lower bound is above 0."""
    d = (a - b).dropna().to_numpy()
    rng = np.random.default_rng(seed)
    boots = rng.choice(d, size=(n, len(d)), replace=True).mean(axis=1)
    alpha = 0.05 / max(n_tests, 1)
    return float(d.mean()), float(np.quantile(boots, alpha / 2)), float(np.quantile(boots, 1 - alpha / 2))


def per_season_ap(df, p, label="made_all_nba") -> pd.Series:
    from sklearn.metrics import average_precision_score
    d = df.loc[p.index].assign(_p=p)
    return d.groupby("season").apply(lambda g: average_precision_score(g[label], g["_p"]), include_groups=False)


def run_all_nba(table, k_map, last_labelled, log=print):
    exact = table["g_counted"].notna().any()
    test = table[table["season"].between(C.TEST_FIRST, last_labelled)]
    rows, fitted, season_ap = [], {}, {}
    for fs in feature_sets(table, log):
        train = table[table["season"].between(fs["first"], C.TRAIN_LAST) & table["made_all_nba"].notna()]
        _, best, oof = _with_blocks(lambda: T.tune(train, "made_all_nba", k_map, log=lambda *_: None,
                                                   grids={"logreg": LR_GRID, "hist_gb": HGB_GRID},
                                                   features=fs["features"]))
        fam = max(best, key=lambda n: (best[n]["cv_pr_auc"], best[n]["cv_precision_at_k"]))
        season_ap[fs["name"]] = per_season_ap(train, oof[fam])
        model = T.fit_calibrated(fam, best[fam]["params"], train, "made_all_nba", oof[fam], fs["features"])
        p = pd.Series(model.predict_proba(test[fs["features"]])[:, 1], index=test.index)
        row = {"set": fs["name"], "n_features": len(fs["features"]), "train_from": fs["first"], "model": fam,
               "cv_pr_auc": best[fam]["cv_pr_auc"], "cv_precision_at_k": best[fam]["cv_precision_at_k"]}
        for tag, ex in (("g", False), ("exact", True)) if exact else (("g", False),):
            met = E.classification_metrics(test, E.apply_eligibility(test, p, exact=ex), "made_all_nba", k_map)
            row.update({f"test_pr_auc_{tag}": met["pr_auc"], f"test_hits_{tag}": met["hits"], "test_slots": met["slots"],
                        f"test_brier_{tag}": met["brier"]})
        rows.append(row)
        fitted[fs["name"]] = {"model": model, "name": fam, "params": best[fam]["params"], "features": fs["features"],
                              "train_seasons": [fs["first"], C.TRAIN_LAST]}
        log(f"[v2] {fs['name']:40s} {fam:8s} CV PR-AUC {row['cv_pr_auc']:.3f} P@k {row['cv_precision_at_k']:.3f} | "
            f"holdout hits {row['test_hits_g']}/{row['test_slots']}")
    res = pd.DataFrame(rows)
    ref = "A base (v1 features)"
    m = len(res) - 1
    stats = [paired_bootstrap(season_ap[s], season_ap[ref], m) for s in res["set"]]
    res["cv_ap_gain_vs_v1"] = [s[0] for s in stats]
    res["gain_ci_low"] = [s[1] for s in stats]
    res["gain_ci_high"] = [s[2] for s in stats]
    res["adopt"] = (res["gain_ci_low"] > 0) & (res["set"] != ref)
    # window-matched test of each group: same training window, with vs without the group
    wref = {n: ("B base (v1 features)" if n in ("B +all Kaggle groups", "B +all Kaggle +onoff") else
                "B +all Kaggle +onoff" if n.startswith("B +all Kaggle +onoff +") or n == "B +everything" else
                ref if n.startswith("A ") and n != ref else None) for n in res["set"]}
    w = [paired_bootstrap(season_ap[n], season_ap[r], m) if r else (np.nan,) * 3 for n, r in wref.items()]
    res["window_ref"] = list(wref.values())
    res["gain_vs_window_ref"] = [x[0] for x in w]
    res["window_ci_low"] = [x[1] for x in w]
    res["window_ci_high"] = [x[2] for x in w]
    res["group_helps_in_window"] = res["window_ci_low"] > 0
    cand = res[res["adopt"]]
    if len(cand):
        top = cand["cv_pr_auc"].max()
        chosen = cand[cand["cv_pr_auc"] >= top - PARSIMONY].sort_values(["n_features", "cv_pr_auc"],
                                                                        ascending=[True, False]).iloc[0]["set"]
        log(f"[v2] chosen All-NBA set: {chosen} (Bonferroni-adjusted per-season CV AP gain CI excludes 0)")
    else:
        chosen = ref
        log("[v2] no feature group improved CV per-season AP with a Bonferroni-adjusted CI above 0; v2 keeps the v1 feature set")
    return res, fitted, chosen


# --------------------------------------------------------------------------- MVP
RANK_CAP = 10  # real-winner rank capped at 10: one atypical MVP (Nash 2005 at #233) must not dominate a mean


def mvp_winner_ranks(df, score) -> pd.Series:
    """Per season: the model's rank of the real MVP, capped at RANK_CAP."""
    out = {}
    for season, g in df.assign(_s=score).groupby("season"):
        order = g.sort_values("_s", ascending=False)
        w = order.index[order["mvp_winner"] == 1]
        if len(w):
            out[season] = min(order.index.get_loc(w[0]) + 1, RANK_CAP)
    return pd.Series(out, dtype=float)


def mvp_eval(df, score) -> dict:
    hits, ranks, top5 = [], [], []
    for _, g in df.assign(_s=score).groupby("season"):
        order = g.sort_values("_s", ascending=False)
        w = order.index[order["mvp_winner"] == 1]
        if len(w) == 0:
            continue
        hits.append(order.index[0] == w[0])
        ranks.append(order.index.get_loc(w[0]) + 1)
        top5.append(int((order.head(5)["mvp_rank"] <= 5).sum()))
    return {"winner_hits": int(np.sum(hits)), "seasons": len(hits),
            "mean_winner_rank_capped": float(np.mean(np.minimum(ranks, RANK_CAP))),
            "median_winner_rank": float(np.median(ranks)), "worst_winner_rank": int(np.max(ranks)),
            "mean_top5_overlap": float(np.mean(top5))}


def mvp_sets(table):
    G, base = C.FEATURE_GROUPS, list(C.BASE_FEATURES)
    kag = G["fatigue"] + G["jump"] + G["teammates"] + G["narrative"]
    sets = [{"name": "A base", "features": base, "first": C.FIRST_SEASON},
            {"name": "A +fatigue", "features": base + G["fatigue"], "first": C.FIRST_SEASON},
            {"name": "A +narrative", "features": base + G["narrative"], "first": C.FIRST_SEASON},
            {"name": "A +all Kaggle groups", "features": base + kag, "first": C.FIRST_SEASON},
            {"name": "B base", "features": base, "first": C.V2_TRAIN_FIRST},
            {"name": "B +all Kaggle +onoff", "features": base + kag + G["onoff"], "first": C.V2_TRAIN_FIRST}]
    usable = [g for g, cols in C.NBA_GROUPS.items()
              if group_coverage(table, cols, C.V2_TRAIN_FIRST) >= C.MIN_GROUP_COVERAGE]
    if usable:
        sets.append({"name": "B +everything", "features": base + kag + G["onoff"] + sum((C.NBA_GROUPS[g] for g in usable), []),
                     "first": C.V2_TRAIN_FIRST})
    return sets


def run_mvp(table, hgb_params, last_labelled, log=print):
    lab = table[table["mvp_share"].notna()]
    test = lab[lab["season"].between(C.TEST_FIRST, last_labelled)]
    rows, fitted, oof_by_set = [], {}, {}
    for fs in mvp_sets(table):
        f = fs["features"]
        oof = []
        for lo, hi in C.V2_CV_BLOCKS:
            tr = lab[lab["season"].between(fs["first"], lo - 1)]
            va = lab[lab["season"].between(lo, hi)]
            m = T.make_model("hist_gb_reg", hgb_params).fit(tr[f], tr["mvp_share"])
            oof.append(pd.Series(m.predict(va[f]), index=va.index))
        oof = pd.concat(oof)
        cvm = mvp_eval(lab.loc[oof.index], oof)
        tr = lab[lab["season"].between(fs["first"], C.TRAIN_LAST)]
        m = T.make_model("hist_gb_reg", hgb_params).fit(tr[f], tr["mvp_share"])
        p = E.apply_eligibility(test, pd.Series(m.predict(test[f]), index=test.index))
        tm = mvp_eval(test, p)
        rows.append({"set": fs["name"], "n_features": len(f), **{f"cv_{k}": v for k, v in cvm.items()},
                     **{f"test_{k}": v for k, v in tm.items()}})
        fitted[fs["name"]] = {"model": m, "name": "hist_gb_reg", "features": f, "train_seasons": [fs["first"], C.TRAIN_LAST]}
        oof_by_set[fs["name"]] = (oof, p)
        log(f"[v2] MVP {fs['name']:28s} CV winners {cvm['winner_hits']}/{cvm['seasons']} (capped mean rank {cvm['mean_winner_rank_capped']:.2f}, worst {cvm['worst_winner_rank']}) "
            f"| holdout {tm['winner_hits']}/{tm['seasons']}")
    res = pd.DataFrame(rows)
    ref = "A base"
    base_r = mvp_winner_ranks(lab.loc[oof_by_set[ref][0].index], oof_by_set[ref][0])
    stats = []
    for s in res["set"]:
        r = mvp_winner_ranks(lab.loc[oof_by_set[s][0].index], oof_by_set[s][0])
        stats.append(paired_bootstrap(base_r, r, len(res) - 1))  # positive = the real winner is ranked higher (better)
    res["cv_winner_rank_gain"] = [s[0] for s in stats]
    res["gain_ci_low"] = [s[1] for s in stats]
    res["gain_ci_high"] = [s[2] for s in stats]
    res["adopt"] = (res["gain_ci_low"] > 0) & (res["set"] != ref)
    wref = {n: ("B base" if n == "B +all Kaggle +onoff" else "B +all Kaggle +onoff" if n == "B +everything" else
                ref if n.startswith("A ") and n != ref else None) for n in res["set"]}
    w = []
    for n, r in wref.items():
        if r is None:
            w.append((np.nan,) * 3)
            continue
        rr = mvp_winner_ranks(lab.loc[oof_by_set[r][0].index], oof_by_set[r][0])
        rn = mvp_winner_ranks(lab.loc[oof_by_set[n][0].index], oof_by_set[n][0])
        w.append(paired_bootstrap(rr, rn, len(res) - 1))
    res["window_ref"] = list(wref.values())
    res["gain_vs_window_ref"] = [x[0] for x in w]
    res["window_ci_low"] = [x[1] for x in w]
    res["group_helps_in_window"] = res["window_ci_low"] > 0
    cand = res[res["adopt"]]
    chosen = (cand.sort_values(["cv_winner_hits", "cv_mean_winner_rank_capped", "n_features"], ascending=[False, True, True])
              .iloc[0]["set"] if len(cand) else ref)
    log(f"[v2] chosen MVP set: {chosen}" + ("" if len(cand) else " (no set beat the base with a CI above 0)"))
    fatigue = fatigue_check(lab, oof_by_set, test)
    fatigue.attrs["cases"] = fatigue_cases(lab, *oof_by_set[ref])
    return res, fitted, chosen, fatigue


def fatigue_cases(lab, oof, p_test) -> pd.DataFrame:
    """Season by season with the base model: reigning MVP, real winner, model's top pick."""
    rows = []
    for part, pred in (("CV", oof), ("holdout", p_test.replace(-np.inf, np.nan))):
        d = lab.loc[pred.index].assign(pred=pred)
        for s, g in d.groupby("season"):
            rei = g[g["mvp_last_season"] == 1]
            win = g[g["mvp_winner"] == 1]
            rows.append({"season": s, "data": part, "reigning_mvp": rei["player"].iloc[0] if len(rei) else "(not qualified)",
                         "winner": win["player"].iloc[0] if len(win) else None,
                         "model_top_pick": g.loc[g["pred"].idxmax(), "player"]})
    out = pd.DataFrame(rows)
    out["reigning_won"] = out["reigning_mvp"] == out["winner"]
    out["model_picked_reigning"] = out["reigning_mvp"] == out["model_top_pick"]
    return out


def fatigue_check(lab, oof_by_set, test) -> pd.DataFrame:
    """Is there MVP voter fatigue? For the reigning MVP: real share minus predicted share (base model vs +fatigue).
    A negative base residual means voters gave reigning MVPs LESS than their stats predict."""
    out = []
    for name in ["A base", "A +fatigue"]:
        if name not in oof_by_set:
            continue
        oof, p_test = oof_by_set[name]
        for part, idx, pred in (("CV 2005-2019", oof.index, oof), ("holdout", p_test.index, p_test)):
            d = lab.loc[idx].assign(pred=pred.replace(-np.inf, 0.0))
            top = d.assign(_r=d.groupby("season")["pred"].rank(ascending=False, method="first"))
            for grp, mask in (("reigning MVP", d["mvp_last_season"] == 1),
                              ("other top-10 by real share", (d["mvp_last_season"] == 0) & (d["mvp_rank"] <= 10))):
                g = d[mask]
                out.append({"model": name, "data": part, "group": grp, "n": len(g),
                            "mean_real_share": g["mvp_share"].mean(), "mean_pred_share": g["pred"].mean(),
                            "mean_residual": (g["mvp_share"] - g["pred"]).mean(),
                            "won": int((g["mvp_winner"] == 1).sum()),
                            "model_top_pick": int((top.loc[g.index, "_r"] == 1).sum())})
    return pd.DataFrame(out)


# --------------------------------------------------------------------------- main
def main(table, label_counts, log=print, out_models=C.MODELS_DIR, out_reports=C.REPORTS_DIR):
    out_models.mkdir(parents=True, exist_ok=True)
    out_reports.mkdir(parents=True, exist_ok=True)
    last_labelled = int(table.loc[table["made_all_nba"].notna(), "season"].max())
    k_map = label_counts["all_nba"]
    res, fitted, chosen = run_all_nba(table, k_map, last_labelled, log)
    res.to_csv(out_reports / "v2_all_nba_sets.csv", index=False)
    if chosen == "A base (v1 features)":  # nothing adopted: serve the exact v1 model, not a refit
        bundle = {**joblib.load(out_models / "all_nba_best.joblib"), "set": "v1 model (no v2 group adopted)"}
    else:
        bundle = {**fitted[chosen], "set": chosen}
    joblib.dump({**bundle, "eligibility_exact": bool(table["g_counted"].notna().any())}, out_models / "all_nba_v2.joblib")

    hgb_params = json.loads((out_reports / "metrics.json").read_text())["all_nba"]["best_by_model"]["hist_gb"]["params"]
    mres, mfitted, mchosen, fatigue = run_mvp(table, hgb_params, last_labelled, log)
    mres.to_csv(out_reports / "v2_mvp_sets.csv", index=False)
    fatigue.to_csv(out_reports / "v2_mvp_fatigue.csv", index=False)
    fatigue.attrs["cases"].to_csv(out_reports / "v2_mvp_fatigue_cases.csv", index=False)
    mb = ({**joblib.load(out_models / "mvp_share.joblib"), "set": "v1 model (no v2 group adopted)"} if mchosen == "A base"
          else {**mfitted[mchosen], "set": mchosen})
    joblib.dump(mb, out_models / "mvp_v2.joblib")

    # SHAP group shares for the chosen All-NBA v2 model
    import explain as X
    groups = dict(X.GROUPS)
    for g, cols in {**C.FEATURE_GROUPS, **C.NBA_GROUPS}.items():
        groups[f"v2: {g}"] = cols
    b = bundle  # explain the model that is actually served
    lab = table[table["made_all_nba"].notna()]
    tr = lab[lab["season"].between(b["train_seasons"][0], C.TRAIN_LAST)]
    te = lab[lab["season"] >= C.TEST_FIRST]
    sv = X.explain_model(b, tr.sample(min(500, len(tr)), random_state=C.RANDOM_STATE), te)
    dfv = pd.DataFrame(sv.values, columns=sv.feature_names)
    gi = pd.Series({g: dfv[[c for c in cols if c in dfv]].sum(axis=1).abs().mean()
                    for g, cols in groups.items() if any(c in dfv for c in cols)})
    gi = (gi / gi.sum()).sort_values(ascending=False)
    gi.to_csv(out_reports / "v2_shap_groups.csv", header=["share"])
    summary = {"all_nba_chosen": chosen, "mvp_chosen": mchosen, "exact_eligibility": bool(table["g_counted"].notna().any()),
               "last_labelled": last_labelled}
    (out_reports / "v2_summary.json").write_text(json.dumps(summary, indent=2))
    return {"all_nba": res, "mvp": mres, "fatigue": fatigue, "shap_groups": gi, **summary}


# --------------------------------------------------------------------------- report
def write_report(r: dict, info: dict | None = None, out_reports=C.REPORTS_DIR):
    fmt = lambda df: df.to_markdown(index=False, floatfmt=".3f")
    a, mv, fat = r["all_nba"], r["mvp"], r["fatigue"]
    n_cv = sum(b - a_ + 1 for a_, b in C.V2_CV_BLOCKS)
    nb = (info or {}).get("nba_api", {})
    L = ["# v2: new feature groups (MVP fatigue, teammates' health, jumps, team with/without, narratives)\n",
         "## What the groups are\n",
         "| group | features | source |", "|---|---|---|"]
    src = {"fatigue": "MVP labels, strictly prior seasons", "jump": "season-relative z this season minus last season",
           "teammates": "Kaggle per-team rows: games missed by rotation teammates (weighted by BPM x minutes), best teammate VORP pace",
           "narrative": "All-Star this season (chosen in February, before voting), team win% change, best player on team, share of team VORP, stat-title leader",
           "onoff": "Basketball-Reference on/off net rating per 100 possessions (1997+)",
           "with_without": "nba_api game logs: team win% in games played vs missed (shrunk by games missed)",
           "availability": "nba_api game logs: games with 20+ min (+ up to two 15-20 min), as share of team games",
           "seed": "nba_api standings: conference seed", "nba_adv": "nba_api advanced dashboard: PIE, net rating",
           "clutch": "nba_api clutch dashboard (last 5 min, within 5 pts)", "est": "nba_api estimated net rating",
           "highlights": "nba_api game logs: triple-doubles and 40-point games per team game"}
    for g, cols in {**C.FEATURE_GROUPS, **C.NBA_GROUPS}.items():
        L.append(f"| {g} | {', '.join(f'`{c}`' for c in cols)} | {src.get(g, '')} |")
    L += ["\nThere is no media/narrative data here, so the narrative features are measurable stand-ins for narrative, not narrative itself.\n",
          "## nba_api data status\n"]
    if nb.get("crosswalk"):
        L.append(f"- Endpoints and seasons cached: { {k: f'{min(v)}-{max(v)} ({len(v)})' for k, v in nb['endpoints'].items()} }")
        L.append(f"- Crosswalk B-Ref -> NBA ids: {nb['crosswalk']} (unmatched rows: `crosswalk_unmatched.csv`).")
    else:
        L.append("- **No nba_api cache yet.** The nba_api groups were skipped (0% coverage). Run `nba_fetch.py`, then rerun the pipeline.")
    L.append("")
    m = len(a) - 1
    L += ["## All-NBA: does each group help?\n",
          f"Every row is tuned and scored on the same forward-chaining CV seasons ({n_cv} seasons, "
          f"{C.V2_CV_BLOCKS[0][0]}-{C.V2_CV_BLOCKS[-1][1]}). `cv_ap_gain_vs_v1` is the mean per-season PR-AUC change vs the v1 "
          f"feature set, with a paired bootstrap CI over seasons, Bonferroni-adjusted for {m} comparisons. A set is adopted only if "
          "the whole CI is above 0. Holdout hits (2020+, 65-game mask) are shown for information and were never used to choose.\n",
          "`gain_vs_window_ref` tests each group fairly inside its own training window (1997+ sets against the 1997+ set without "
          "that group). `adopt` is the production bar: the set must also beat the 1980+ v1 model.\n",
          fmt(a.drop(columns=[c for c in a.columns if c.startswith("test_brier")])), "\n"]
    helps = a[a["group_helps_in_window"] & ~a["adopt"]]
    if len(helps):
        L.append(f"Groups that help inside their window but do not beat the v1 model overall: {', '.join(helps['set'])}.\n")
    adopted = a[a["adopt"]]
    L.append(f"**Verdict:** chosen v2 All-NBA set = **{r['all_nba_chosen']}**. "
             + ("No new group improved All-NBA prediction beyond noise. The v1 features (box-score impact, scoring, availability, "
                "team success) already capture what these groups add."
                if adopted.empty else f"Groups that passed: {', '.join(adopted['set'])}.") + "\n")
    L += ["## MVP: does each group help?\n",
          f"HistGradientBoosting regression on vote share. `cv_winner_rank_gain` = how many places higher the real winner is ranked "
          f"than with the base features (rank capped at {RANK_CAP}, mean over {n_cv} CV seasons, Bonferroni-adjusted bootstrap CI). "
          "The cap matters: 1997+ sets train their first CV block on only 8 seasons, and that model ranks Steve Nash's "
          "atypical 2005-06 MVPs in the 200s; `worst_winner_rank` shows these outliers.\n",
          fmt(mv), "\n",
          f"**Verdict:** chosen v2 MVP set = **{r['mvp_chosen']}**"
          + (" (nothing passed the adjusted CI; differences of one winner in 15 seasons are noise)." if not mv["adopt"].any()
             else ".") + "\n",
          "## MVP voter fatigue: is there any?\n",
          "Reigning MVPs compared with their predicted share. A **negative** residual (real share below predicted) would mean "
          "fatigue. `won` vs `model_top_pick` compares how often reigning MVPs actually won with how often the model ranked them first.\n",
          fmt(fat), "\n"]
    base_cv = fat[(fat["model"] == "A base") & (fat["data"] == "CV 2005-2019") & (fat["group"] == "reigning MVP")]
    if len(base_cv):
        res_ = float(base_cv["mean_residual"].iloc[0])
        won, pick = int(base_cv["won"].iloc[0]), int(base_cv["model_top_pick"].iloc[0])
        verdict = ("**No evidence of voter fatigue.** Reigning MVPs received *more* vote share than their stats predict "
                   f"(mean residual {res_:+.3f}), and they won {won} times vs {pick} model top picks."
                   if res_ >= 0 and won >= pick else
                   f"**Some evidence of fatigue:** residual {res_:+.3f}; won {won} vs {pick} model top picks.")
        L.append(verdict + f" Small samples: {int(base_cv['n'].iloc[0])} reigning-MVP seasons in CV.\n")
    cases = fat.attrs.get("cases")
    if cases is not None and len(cases):
        lost_picked = cases[cases.model_picked_reigning & ~cases.reigning_won]
        won_unpicked = cases[~cases.model_picked_reigning & cases.reigning_won]
        L += ["Season by season (base model):\n", fmt(cases), "\n",
              f"Reigning MVPs the model picked who then lost (the fatigue-looking cases): "
              f"{', '.join(f'{r.reigning_mvp} {r.season} (won by {r.winner})' for r in lost_picked.itertuples()) or 'none'}. "
              f"Reigning MVPs the model did not pick who won anyway: "
              f"{', '.join(f'{r.reigning_mvp} {r.season}' for r in won_unpicked.itertuples()) or 'none'}. "
              + (f"Both kinds occur ({len(lost_picked)} vs {len(won_unpicked)}), so there is no consistent fatigue pattern.\n"
               if abs(len(lost_picked) - len(won_unpicked)) <= 1 else
               f"The fatigue-looking cases outnumber the others ({len(lost_picked)} vs {len(won_unpicked)}), "
               "a hint of fatigue in small samples.\n" if len(lost_picked) > len(won_unpicked) else
               f"Reigning MVPs won unexpectedly more often ({len(won_unpicked)} vs {len(lost_picked)}): "
               "the opposite of fatigue.\n")]
    L += ["## What the chosen v2 All-NBA model relies on (share of mean |SHAP| by group, holdout)\n",
          r["shap_groups"].rename("share").reset_index().rename(columns={"index": "group"}).to_markdown(index=False, floatfmt=".3f"),
          "\n"]
    (out_reports / "v2_report.md").write_text("\n".join(L) + "\n")
    print(f"[v2] wrote {out_reports / 'v2_report.md'}")
