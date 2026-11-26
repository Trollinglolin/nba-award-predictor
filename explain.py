"""SHAP explanations for the All-NBA models.

- Chosen model (by CV): explained with the matching explainer (LinearExplainer for the logistic pipeline,
  TreeExplainer for tree models). The logistic model is always explained too, as a cross-check against its coefficients.
- The best tree model is also explained so the two model families can be compared.
- SHAP values are in the base model's log-odds (before Platt calibration, which is monotone).
- Outputs: reports/figures/*.png and reports/explain.md.
"""
from __future__ import annotations

import json

import matplotlib
matplotlib.use("Agg")
import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from scipy.stats import spearmanr

import config as C
import evaluate as E

GROUPS = {
    "impact metrics (BPM/VORP/WS/PER)": ["z_per", "z_ws", "z_ws_48", "z_bpm", "z_obpm", "z_dbpm", "z_vorp", "z_ows",
                                         "z_dws", "rank_vorp", "rank_ws", "rank_bpm", "rank_per"],
    "box-score volume (pts/reb/ast/...)": ["z_pts_per_game", "z_trb_per_game", "z_ast_per_game", "z_stl_per_game",
                                           "z_blk_per_game", "rank_pts_per_game", "z_usg_percent", "z_ts_percent"],
    "availability (games, minutes)": ["g_share", "z_mp_per_game", "traded"],
    "team success": ["team_win_pct", "team_srs", "team_n_rtg", "team_win_rank", "team_playoffs"],
    "voter history (prior selections)": ["prior_all_nba", "prev_all_nba", "prior_all_star", "prior_mvp_top5"],
    "age": ["age"],
}


def explain_model(bundle: dict, X_bg: pd.DataFrame, X: pd.DataFrame) -> shap.Explanation:
    base = bundle["model"].base if isinstance(bundle["model"], E.PlattCalibrated) else bundle["model"]
    feats = bundle["features"]
    if bundle["name"] == "logreg":
        prep = base[:-1]  # imputer + scaler
        lr = base[-1]
        e = shap.LinearExplainer(lr, prep.transform(X_bg[feats]))
        sv = e(prep.transform(X[feats]))
        vals, base_vals = sv.values, sv.base_values
    else:
        model = base[-1] if hasattr(base, "steps") else base
        Xt = base[:-1].transform(X[feats]) if hasattr(base, "steps") else X[feats]
        sv = shap.TreeExplainer(model)(Xt)
        vals, base_vals = sv.values, sv.base_values
        if vals.ndim == 3:  # random forest: (n, features, classes) -> class 1
            vals, base_vals = vals[:, :, 1], base_vals[:, 1]
    return shap.Explanation(values=vals, base_values=base_vals, data=X[feats].to_numpy(), feature_names=feats)


def group_importance(sv: shap.Explanation) -> pd.Series:
    df = pd.DataFrame(sv.values, columns=sv.feature_names)
    out = {g: df[[c for c in cols if c in df]].sum(axis=1).abs().mean() for g, cols in GROUPS.items()}
    return pd.Series(out).sort_values(ascending=False)


def _save(fig_path):
    plt.tight_layout()
    plt.savefig(fig_path, dpi=130, bbox_inches="tight")
    plt.close("all")


def story(sv: shap.Explanation, i: int, row: pd.Series, n=4) -> str:
    contrib = pd.Series(sv.values[i], index=sv.feature_names).sort_values()
    fmt = lambda items: ", ".join(f"`{f}` = {row[f]:.2f} ({v:+.2f})" for f, v in items)
    up = [(f, v) for f, v in contrib[::-1].head(n).items() if v > 0]
    down = [(f, v) for f, v in contrib.head(n).items() if v < 0]
    return f"pushed up by {fmt(up) or 'nothing'}; pushed down by {fmt(down) or 'nothing'}"


def main(models_dir=C.MODELS_DIR, reports_dir=C.REPORTS_DIR, table: pd.DataFrame | None = None, log=print):
    fig_dir = reports_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    if table is None:
        table = pd.read_csv(C.TABLE_PATH)
    metrics = json.loads((reports_dir / "metrics.json").read_text())["all_nba"]
    chosen = metrics["chosen_model"]
    best = metrics["best_by_model"]
    tree_best = max([n for n in best if n != "logreg"], key=lambda n: best[n]["cv_pr_auc"])

    lab = table[table["made_all_nba"].notna()]
    train = lab[lab["season"].between(C.FIRST_SEASON, C.TRAIN_LAST)]
    test = lab[lab["season"] >= C.TEST_FIRST].reset_index(drop=True)
    bg = train.sample(min(500, len(train)), random_state=C.RANDOM_STATE)

    names = list(dict.fromkeys([chosen, "logreg", tree_best]))
    expl, bundles = {}, {}
    for name in names:
        bundles[name] = joblib.load(models_dir / f"all_nba_{name}.joblib")
        expl[name] = explain_model(bundles[name], bg, test)
        plt.figure()
        shap.plots.beeswarm(expl[name], max_display=15, show=False)
        plt.title(f"SHAP summary: {name} (All-NBA, test seasons {C.TEST_FIRST}+)")
        _save(fig_dir / f"shap_summary_{name}.png")
        log(f"[explain] SHAP computed for {name}")

    sv = expl[chosen]
    imp = {n: pd.Series(np.abs(e.values).mean(0), index=e.feature_names).sort_values(ascending=False) for n, e in expl.items()}
    top3 = list(imp[chosen].index[:3])
    for f in top3:
        plt.figure()
        shap.plots.scatter(sv[:, f], color=sv[:, "team_win_pct"] if f != "team_win_pct" else sv[:, "z_vorp"], show=False)
        plt.title(f"SHAP dependence: {f} ({chosen})")
        _save(fig_dir / f"shap_dependence_{f}.png")

    # ---- three player stories from the chosen model (65-game mask applied for 2024+)
    bundle = bundles[chosen]
    p = pd.Series(bundle["model"].predict_proba(test[bundle["features"]])[:, 1], index=test.index)
    pm = E.apply_eligibility(test, p)
    rank = pm.groupby(test["season"]).rank(ascending=False, method="min")
    k_map = test.groupby("season")["made_all_nba"].sum().astype(int).to_dict()
    picked = E.top_k(test, pm, k_map)
    cases = {
        "correct pick": test[(test["made_all_nba"] == 1) & picked & (test["season"] == test["season"].max())].index[:1],
        "surprising miss": rank[(test["made_all_nba"] == 1) & ~picked].sort_values(ascending=False).index[:1],
        "false positive": pm[(test["made_all_nba"] == 0) & picked].sort_values(ascending=False).index[:1],
    }
    stories = []
    for label, idx in cases.items():
        if len(idx) == 0:
            continue
        i = int(idx[0])
        r = test.loc[i]
        plt.figure()
        shap.plots.waterfall(sv[i], max_display=12, show=False)
        plt.title(f"{label}: {r['player']} {int(r['season'])} (p={p[i]:.2f}, rank {int(rank[i])})")
        fname = f"shap_waterfall_{label.replace(' ', '_')}.png"
        _save(fig_dir / fname)
        stories.append((label, r, p[i], int(rank[i]), story(sv, i, r), fname))

    # ---- cross-check: logistic coefficients vs LinearExplainer
    lr_pipe = bundles["logreg"]["model"].base
    coef = pd.Series(lr_pipe[-1].coef_[0], index=bundles["logreg"]["features"])
    rho = spearmanr(coef.abs(), imp["logreg"].reindex(coef.index)).statistic
    sign_ok = np.mean([np.sign(coef[f]) == np.sign(np.corrcoef(expl["logreg"][:, f].data, expl["logreg"][:, f].values)[0, 1])
                       for f in coef.index if np.std(expl["logreg"][:, f].data) > 0])
    rho_models = spearmanr(imp["logreg"], imp[tree_best].reindex(imp["logreg"].index)).statistic

    # ---- write-up
    gi = {n: group_importance(e) for n, e in expl.items()}
    lines = [f"# SHAP explanations (All-NBA)\n",
             f"Explained on the {len(test)} qualified player-seasons of the holdout ({C.TEST_FIRST}-{int(test.season.max())}). "
             "Values are contributions to the model's log-odds (before the monotone Platt calibration).\n",
             f"Chosen model (by forward-chaining CV PR-AUC): **{chosen}**; best tree model: **{tree_best}**.\n",
             "## Feature groups (share of total mean |SHAP|, summed within group)\n",
             "Each column is normalised to sum to 1. The logistic model is in log-odds; TreeExplainer on a random forest "
             "is in probability units, so only the shares are comparable.\n",
             pd.DataFrame({n: g / g.sum() for n, g in gi.items()}).round(3).sort_values(chosen, ascending=False).to_markdown(), "\n",
             "## Top 12 single features (mean |SHAP|)\n",
             pd.DataFrame({n: imp[n].head(12).index for n in imp}).to_markdown(), "\n",
             "## Cross-checks\n",
             f"- Logistic model: Spearman between |coefficient| (standardised) and mean |SHAP| = **{rho:.2f}**; "
             f"coefficient sign matches the SHAP-vs-value direction for **{sign_ok:.0%}** of features.",
             f"- Agreement between the logistic and {tree_best} importance rankings (Spearman) = **{rho_models:.2f}**.",
             "- Many inputs are strongly correlated (WS, OWS, VORP, BPM, PER all measure overall impact), so the credit is "
             "shared among them unpredictably. In the logistic model some of them even get a negative sign "
             "(for example `z_bpm`) once the others are included. Group totals are more stable than single-feature ranks.\n",
             "## Three players\n",
             "Contributions are measured against an *average qualified player* (the waterfall's E[f(X)]), not against the "
             "15th-best player. An All-NBA candidate can be far above average on everything and still miss the cut. "
             "The p shown is the calibrated probability.\n"]
    for label, r, prob, rk, txt, fname in stories:
        lines.append(f"- **{label}: {r['player']} {int(r['season'])}** (actual: {'All-NBA ' + str(int(r['all_nba_team'])) + ' team' if r['made_all_nba'] else 'not selected'}; "
                     f"{int(r['g'])} games, VORP {r['vorp']}, team win% {r['team_win_pct']:.3f}) -> p = {prob:.2f}, rank {rk}. "
                     f"Largest contributions: {txt}. Plot: `figures/{fname}`")
    (reports_dir / "explain.md").write_text("\n".join(lines) + "\n")
    log(f"[explain] wrote {reports_dir / 'explain.md'} and figures in {fig_dir}")
    return {"chosen": chosen, "importance": imp, "groups": gi, "rho_coef_shap": rho, "stories": stories}


if __name__ == "__main__":
    main()
