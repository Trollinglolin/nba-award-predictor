"""v3: pre-registered feature-set tests from reports/improvement_plan_2024_25.md.

Same harness as v2 (experiments.py): every set is tuned on the shared forward-chaining CV blocks (2005-2019) with the
FROZEN v2 grids, scored by mean per-season PR-AUC gain vs A base with a paired season bootstrap, Bonferroni-adjusted
for the 8 eligible sets. A set is adopted only if its whole CI is above 0 against A base (B-window sets also report
their in-window gain vs B base, as a diagnostic). The 2020+ holdout is reported, never used to choose, and is
contaminated for H6/H7/H8 (designed after looking at it). The clean test is 2025-26 once its labels are published.

H6c (finisher-big cap for positionless seasons) is a fixed rule, not a feature set: it is scored on the holdout for
information and on the 2026 labels as the real test.

    .venv/bin/python -u -W ignore experiments_v3.py
"""
from __future__ import annotations

import json
import warnings

import joblib
import numpy as np
import pandas as pd

import config as C
import evaluate as E
import experiments as X
import features as F
import train as T

warnings.filterwarnings("ignore")

N_TESTS = 8  # H1a, H1b, H1c, H3, H4, H6a, H7, H8
H1C_LR_GRID = ([{"C": c} for c in (0.003, 0.01, 0.03, 0.1, 0.3, 1.0)]
               + [{"C": c, "l1_ratio": r, "solver": "saga"} for c in (0.03, 0.1, 0.3) for r in (0.2, 0.5)])
FROZEN = {"logreg": X.LR_GRID, "hist_gb": X.HGB_GRID}


def v3_sets(family: str | None = None) -> list[dict]:
    base, G = list(C.BASE_FEATURES), C.V3_GROUPS
    A, B = C.FIRST_SEASON, C.V2_TRAIN_FIRST
    s = [
        {"name": "A base (v1 features)", "features": base, "first": A, "ref": None},
        {"name": "B base (v1 features)", "features": base, "first": B, "ref": None},
        {"name": "H1a team block", "features": [f for f in base if f not in C.V3_TEAM_BLOCK_DROP], "first": A},
        {"name": "H1b pts curvature", "features": [f for f in base if f != "rank_pts_per_game"] + ["z_pts_per_game_sq"],
         "first": A},
        {"name": "H1c wider regularisation", "features": base, "first": A,
         "grids": {"logreg": H1C_LR_GRID, "hist_gb": X.HGB_GRID}},
        {"name": "H3 team role", "features": base + G["H3_role"], "first": A},
        {"name": "H4 seed", "features": base + G["H4_seed"], "first": B},
        {"name": "H6a archetype", "features": base + G["H6a_archetype"], "first": B},
        {"name": "H7 defense gate", "features": base + G["H7_defense"], "first": A},
        {"name": "H8 improver", "features": base + G["H8_improver"], "first": A},
    ]
    for d in s:
        d.setdefault("ref", "B base (v1 features)" if d["first"] == B else "A base (v1 features)")
        if family:  # same model family for every set, so a gain is the features' alone
            g = d.get("grids", FROZEN)
            d["grids"] = {family: g[family]}
    return s


def fit_set(table, fs, k_map, test, exact):
    train = table[table["season"].between(fs["first"], C.TRAIN_LAST) & table["made_all_nba"].notna()]
    cov = float(train[fs["features"]].notna().all(axis=1).mean())
    _, best, oof = X._with_blocks(lambda: T.tune(train, "made_all_nba", k_map, log=lambda *_: None,
                                                 grids=fs.get("grids", FROZEN), features=fs["features"]))
    fam = max(best, key=lambda n: (best[n]["cv_pr_auc"], best[n]["cv_precision_at_k"]))
    model = T.fit_calibrated(fam, best[fam]["params"], train, "made_all_nba", oof[fam], fs["features"])
    p = pd.Series(model.predict_proba(test[fs["features"]])[:, 1], index=test.index)
    met = E.classification_metrics(test, E.apply_eligibility(test, p, exact=exact), "made_all_nba", k_map)
    ps = E.per_season_hits(test, E.apply_eligibility(test, p, exact=exact), "made_all_nba", k_map)
    row = {"set": fs["name"], "n_features": len(fs["features"]), "train_from": fs["first"], "model": fam,
           "params": json.dumps(best[fam]["params"]), "train_coverage": round(cov, 3),
           "cv_pr_auc": best[fam]["cv_pr_auc"], "cv_precision_at_k": best[fam]["cv_precision_at_k"],
           "test_pr_auc": met["pr_auc"], "test_hits": met["hits"], "test_slots": met["slots"],
           **{f"hits_{int(r.season)}": int(r.hits) for r in ps.itertuples()}}
    bundle = {"model": model, "name": fam, "params": best[fam]["params"], "features": fs["features"],
              "train_seasons": [fs["first"], C.TRAIN_LAST]}
    return row, X.per_season_ap(train, oof[fam]), bundle, p


def h6c_check(table, bundle, k_map, test, exact) -> pd.DataFrame:
    """Score the holdout with and without the finisher-big rule (information only: 2024-25 informed the rule)."""
    rows = []
    for name, df in (("served model", test), ("served model + H6c rule", F.finisher_big_rule(test))):
        p = pd.Series(bundle["model"].predict_proba(df[bundle["features"]])[:, 1], index=df.index)
        ps = E.per_season_hits(df, E.apply_eligibility(df, p, exact=exact), "made_all_nba", k_map)
        rows.append({"scorer": name, **{f"hits_{int(r.season)}": int(r.hits) for r in ps.itertuples()},
                     "total": int(ps["hits"].sum())})
    return pd.DataFrame(rows)


def main(table=None, label_counts=None, log=print, out_models=C.MODELS_DIR, out_reports=C.REPORTS_DIR, family=None,
         n_tests=N_TESTS):
    """family=None: best of logreg/hist_gb per set (v2 harness). family="logreg": every set and base is logistic
    regression. Both passes were run, so the report uses n_tests=16 for adoption (8 sets x 2 passes)."""
    if table is None:
        table = pd.read_csv(C.TABLE_PATH, low_memory=False)
    if label_counts is None:
        L = F.label_tables(F.load_raw())
        label_counts = {"all_nba": L["all_nba"].groupby("season").size().to_dict()}
    k_map = label_counts["all_nba"]
    last = int(table.loc[table["made_all_nba"].notna(), "season"].max())
    test = table[table["season"].between(C.TEST_FIRST, last) & table["made_all_nba"].notna()]
    exact = bool(table["g_counted"].notna().any())
    rows, ap, bundles, preds = [], {}, {}, {}
    for fs in v3_sets(family):
        row, ap[fs["name"]], bundles[fs["name"]], preds[fs["name"]] = fit_set(table, fs, k_map, test, exact)
        row["window_ref"] = fs["ref"]
        rows.append(row)
        log(f"[v3] {fs['name']:28s} {row['model']:8s} cov {row['train_coverage']:.2f} CV PR-AUC {row['cv_pr_auc']:.4f} "
            f"| holdout hits {row['test_hits']}/{row['test_slots']}")
    res = pd.DataFrame(rows)
    ref = "A base (v1 features)"
    st = [X.paired_bootstrap(ap[s], ap[ref], n_tests) for s in res["set"]]
    res["gain_vs_A"], res["ci_low"], res["ci_high"] = zip(*st)
    w = [X.paired_bootstrap(ap[s], ap[r], n_tests) if isinstance(r, str) else (np.nan,) * 3
         for s, r in zip(res["set"], res["window_ref"])]
    res["gain_vs_window"], res["window_ci_low"], res["window_ci_high"] = zip(*w)
    eligible = ~res["set"].str.contains("base")
    res["adopt"] = eligible & (res["ci_low"] > 0)

    # confirmatory combo of every adopted set (not part of the Bonferroni family)
    adopted = res.loc[res["adopt"], "set"].tolist()
    chosen = ref
    if adopted:
        sets = {s["name"]: s for s in v3_sets(family)}
        feats = list(dict.fromkeys(sum((sets[n]["features"] for n in adopted), [])))
        first = max(sets[n]["first"] for n in adopted)
        if len(adopted) > 1:
            fs = {"name": "combo of adopted", "features": feats, "first": first, "grids": sets[adopted[0]].get("grids", FROZEN)}
            row, ap[fs["name"]], bundles[fs["name"]], preds[fs["name"]] = fit_set(table, fs, k_map, test, exact)
            g = X.paired_bootstrap(ap[fs["name"]], ap[ref], 1)
            row.update({"gain_vs_A": g[0], "ci_low": g[1], "ci_high": g[2], "adopt": g[1] > 0, "window_ref": ref})
            res = pd.concat([res, pd.DataFrame([row])], ignore_index=True)
            log(f"[v3] combo CV gain {g[0]:+.4f} CI [{g[1]:+.4f}, {g[2]:+.4f}]")
        cand = res[res["adopt"]]
        top = cand["cv_pr_auc"].max()
        chosen = cand[cand["cv_pr_auc"] >= top - X.PARSIMONY].sort_values(["n_features", "cv_pr_auc"],
                                                                          ascending=[True, False]).iloc[0]["set"]
    log(f"[v3] adopted sets: {adopted or 'none'}; served: {chosen}")
    res.to_csv(out_reports / "v3_all_nba_sets.csv", index=False)

    if chosen == ref:  # nothing adopted: serve the exact v1 model, not a refit
        bundle = {**joblib.load(out_models / "all_nba_best.joblib"), "set": "v1 model (no v3 set adopted)"}
    else:
        bundle = {**bundles[chosen], "set": chosen}
    joblib.dump({**bundle, "eligibility_exact": exact}, out_models / "all_nba_v3.joblib")
    h6c = h6c_check(table, bundle, k_map, test, exact)
    h6c.to_csv(out_reports / "v3_h6c_holdout.csv", index=False)

    # diagnostics: named players and centers in the predicted top 15 (holdout, never used to choose)
    diag = []
    for name in res["set"]:
        p = E.apply_eligibility(test, preds[name], exact=exact)
        rk = p.groupby(test["season"]).rank(ascending=False, method="first")
        top = test[rk <= 15]
        r25 = test.assign(rk=rk)[test["season"] == 2025].set_index("player")["rk"]
        diag.append({"set": name, "rank_mitchell_2025": r25.get("Donovan Mitchell"),
                     "rank_jwilliams_2025": r25.get("Jalen Williams"),
                     "rank_morant_2022": test.assign(rk=rk).query("season == 2022 and player == 'Ja Morant'")["rk"].min(),
                     "pred_C_2024_25": int((top["season"] >= 2024).mul(top["pos_group"] == "C").sum()),
                     "actual_C_2024_25": int(((test["season"] >= 2024) & (test["pos_group"] == "C")
                                              & (test["made_all_nba"] == 1)).sum()),
                     "holdout_misses": int(((rk > 15) & (test["made_all_nba"] == 1)).sum())})
    diag = pd.DataFrame(diag)
    diag.to_csv(out_reports / "v3_diagnostics.csv", index=False)
    write_report(res, chosen, h6c, diag, out_reports, n_tests, family)
    return res, chosen


def write_report(res, chosen, h6c, diag, out_reports=C.REPORTS_DIR, n_tests=N_TESTS, family=None):
    cols = ["set", "n_features", "train_from", "model", "train_coverage", "cv_pr_auc", "gain_vs_A", "ci_low", "ci_high",
            "adopt", "window_ref", "gain_vs_window", "window_ci_low", "window_ci_high", "test_hits", "test_slots"]
    hit_cols = [c for c in res.columns if c.startswith("hits_")]
    L = [f"# v3: pre-registered feature sets (improvement plan 2024-25) - model family: {family or 'best of logreg/hist_gb'}\n",
         "Same harness as v2. Every set is tuned on the forward-chaining CV seasons 2005-2019 with the frozen v2 grids "
         "(H1c alone gets a wider logistic grid). `gain_vs_A` is the mean per-season PR-AUC change vs the 1980+ v1 "
         f"feature set, with a paired season bootstrap CI Bonferroni-adjusted for {n_tests} comparisons (8 sets x 2 passes: best family, and logistic regression only). **A set is adopted "
         "only if its whole CI is above 0.** 1997+ sets also show their gain vs B base (same window) as a diagnostic.\n",
         "Holdout hits (2020-2025, 65-game mask) are shown for information only. H6-H8 were designed after looking at "
         "those seasons, so their holdout numbers are optimistic. The clean test is 2025-26.\n",
         res[cols].to_markdown(index=False, floatfmt=".4f"), "\n",
         "## Holdout hits per season\n", res[["set"] + hit_cols].to_markdown(index=False), "\n",
         f"**Served model:** {chosen}.\n",
         "## Diagnostics (holdout; never used to choose)\n", diag.to_markdown(index=False), "\n",
         "## H6c finisher-big rule (fixed, positionless seasons only)\n",
         "A Rim-Running Big listed PF/C who is not his team's leading scorer has his rebounding, blocks, DWS and TS% "
         "z-scores capped at 0 (league average) in seasons from 2024. No parameter was tuned. 2024-25 informed the "
         "rule, so this table cannot validate it; only 2025-26 can.\n", h6c.to_markdown(index=False), "\n"]
    (out_reports / "v3_report.md").write_text("\n".join(L))


CANDIDATE_SPEC = """Opt-in v3 candidate, fixed 2026-11-15 from three hypotheses, NOT validated by CV.
- Features: v1 features + H7 defense gate (def_pos, def_neg, offensive_ace, def_neg_x_ace)
            + H8 improver (team_scoring_leader, lead_improver).
- Model: logistic regression, C chosen on the frozen v2 grid by forward-chaining CV 2005-2019, trained 1980-2019,
  Platt-calibrated on the out-of-fold predictions.
- Scoring rule H6c, seasons >= 2024 only: a Rim-Running Big listed PF/C who is not his team's leading scorer has
  z_trb_per_game, z_blk_per_game, z_dws and z_ts_percent capped at 0 before scoring.
- Test: 2025-26 labels, scored once against v1. +/-1 hit out of 15 is noise."""


def build_candidate(table=None, label_counts=None, log=print, out_models=C.MODELS_DIR, out_reports=C.REPORTS_DIR):
    if table is None:
        table = pd.read_csv(C.TABLE_PATH, low_memory=False)
    if label_counts is None:
        L = F.label_tables(F.load_raw())
        label_counts = {"all_nba": L["all_nba"].groupby("season").size().to_dict()}
    k_map = label_counts["all_nba"]
    last = int(table.loc[table["made_all_nba"].notna(), "season"].max())
    test = table[table["season"].between(C.TEST_FIRST, last) & table["made_all_nba"].notna()]
    exact = bool(table["g_counted"].notna().any())
    feats = list(C.BASE_FEATURES) + C.V3_GROUPS["H7_defense"] + C.V3_GROUPS["H8_improver"]
    base = {"name": "A base (v1 features)", "features": list(C.BASE_FEATURES), "first": C.FIRST_SEASON,
            "grids": {"logreg": X.LR_GRID}}
    cand = {"name": "v3 candidate", "features": feats, "first": C.FIRST_SEASON, "grids": {"logreg": X.LR_GRID}}
    rb, ap_b, _, _ = fit_set(table, base, k_map, test, exact)
    rc, ap_c, bundle, _ = fit_set(table, cand, k_map, test, exact)
    g = X.paired_bootstrap(ap_c, ap_b, 1)
    v1 = joblib.load(out_models / "all_nba_best.joblib")
    hits = {}
    for name, b, rule in (("v1 (served)", v1, False), ("candidate, no rule", bundle, False), ("candidate + H6c rule", bundle, True)):
        df = F.finisher_big_rule(test) if rule else test
        p = pd.Series(b["model"].predict_proba(df[b["features"]])[:, 1], index=df.index)
        ps = E.per_season_hits(df, E.apply_eligibility(df, p, exact=exact), "made_all_nba", k_map)
        hits[name] = {**{int(r.season): int(r.hits) for r in ps.itertuples()}, "total": int(ps["hits"].sum())}
    hits = pd.DataFrame(hits).T
    joblib.dump({**bundle, "set": "v3 candidate (opt-in)", "spec": CANDIDATE_SPEC, "rule": "finisher_big_rule",
                 "eligibility_exact": exact}, out_models / "all_nba_v3_candidate.joblib")
    txt = ["# v3 opt-in candidate\n", "```", CANDIDATE_SPEC, "```\n",
           f"CV (2005-2019) per-season PR-AUC gain vs v1 features, same family and grid: **{g[0]:+.4f}**, "
           f"95% CI [{g[1]:+.4f}, {g[2]:+.4f}] (single uncorrected comparison; exploratory). "
           f"Tuned C: {rc['params']}.\n",
           "Holdout hits (2020-2025, 65-game mask). H7, H8 and H6c were designed after looking at these seasons, so "
           "the candidate's numbers here are optimistic and cannot validate it:\n", hits.to_markdown(), "\n"]
    (out_reports / "v3_candidate.md").write_text("\n".join(txt))
    log(f"[v3] candidate CV gain {g[0]:+.4f} CI [{g[1]:+.4f}, {g[2]:+.4f}]\n{hits.to_string()}")
    return bundle, hits


if __name__ == "__main__":
    import sys
    if sys.argv[1:] == ["candidate"]:
        build_candidate()
        raise SystemExit
    fam = sys.argv[1] if len(sys.argv) > 1 else None
    out = C.REPORTS_DIR / "v3_logreg" if fam else C.REPORTS_DIR
    out.mkdir(exist_ok=True)
    main(family=fam, n_tests=16, out_reports=out)
