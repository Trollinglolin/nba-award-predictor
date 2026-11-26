"""Assemble reports/report.md and summary charts from the training outputs (no model fitting here)."""
from __future__ import annotations

import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

import config as C

COLORS = {"model": "#2a6fdb", "vorp": "#e8833a", "ws": "#9aa5b1", "pts": "#c9cfd6"}


def charts(reports=C.REPORTS_DIR):
    fig_dir = reports / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    m = json.loads((reports / "metrics.json").read_text())["all_nba"]
    chosen = m["chosen_model"]
    ps = pd.read_csv(reports / "test_per_season.csv")
    ps = ps[ps["variant"] == "eligibility_mask"]
    piv = ps.pivot(index="season", columns="scorer", values="hits")
    k = ps.groupby("season")["k"].first()
    series = [(chosen, COLORS["model"]), ("rank_by_vorp", COLORS["vorp"]), ("rank_by_ws", COLORS["ws"]),
              ("rank_by_pts", COLORS["pts"])]
    fig, ax = plt.subplots(figsize=(8, 3.8))
    w = 0.2
    for i, (name, col) in enumerate(series):
        ax.bar(piv.index + (i - 1.5) * w, piv[name], width=w, color=col, label=name)
    ax.plot(k.index, k.values, "k--", lw=1, label="slots (k)")
    ax.set_ylim(0, 16)
    ax.set_ylabel("real All-NBA players in predicted top-k")
    ax.set_title(f"Holdout per season: {chosen} vs baselines (65-game mask from 2024)")
    ax.legend(frameon=False, ncol=5, fontsize=8, loc="lower center", bbox_to_anchor=(0.5, -0.3))
    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()
    plt.savefig(fig_dir / "holdout_per_season.png", dpi=130, bbox_inches="tight")
    plt.close(fig)

    cal = pd.read_csv(reports / "calibration.csv")
    fig, ax = plt.subplots(figsize=(4.2, 4))
    ax.plot([0, 1], [0, 1], color="#9aa5b1", lw=1)
    ax.scatter(cal["mean_pred"], cal["observed"], s=cal["n"].clip(upper=400) / 3 + 10, color=COLORS["model"])
    for r in cal.itertuples():
        ax.annotate(f"n={r.n}", (r.mean_pred, r.observed), fontsize=7, xytext=(4, -8), textcoords="offset points")
    ax.set_xlabel("predicted probability (calibrated)")
    ax.set_ylabel("observed selection rate")
    ax.set_title(f"Calibration on holdout ({chosen})")
    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()
    plt.savefig(fig_dir / "calibration.png", dpi=130, bbox_inches="tight")
    plt.close(fig)


def _fmt(df):
    return df.to_markdown(index=False, floatfmt=".3f")


def cv_brier(name, params, reports=C.REPORTS_DIR) -> float:
    """Uncalibrated forward-chaining CV Brier of the chosen configuration."""
    r = pd.read_csv(reports / "cv_results.csv")
    return float(r[(r["model"] == name) & (r["params"] == json.dumps(params))]["cv_brier"].iloc[0])


def verdicts(m, chosen, mk, ps, posi, preds, cv) -> dict:
    """Every conclusion sentence in the report is computed here from the numbers, so a data refresh cannot leave a stale claim."""
    out = {}
    seasons = ps["season"].tolist()
    n_s, slots = len(seasons), int(mk.loc[chosen, "slots"])
    late = [s for s in seasons if s >= C.POSITIONLESS_FROM]
    out["sizes"] = (f"{n_s} holdout seasons ({min(seasons)}-{max(seasons)}), {slots} slots; "
                    f"{len(late)} of them ({len(late) * 15} slots) are in the positionless / 65-game era")
    n_cv = sum(b - a + 1 for a, b in C.CV_BLOCKS)
    cv_rng = f"{C.CV_BLOCKS[0][0]}-{C.CV_BLOCKS[-1][1]}"
    d = (ps[chosen] - ps["rank_by_vorp"])
    wins, losses, ties = int((d > 0).sum()), int((d < 0).sum()), int((d == 0).sum())
    dh = int(mk.loc[chosen, "hits"] - mk.loc["rank_by_vorp", "hits"])
    dcv = m["best_by_model"][chosen]["cv_precision_at_k"] - m["baselines_cv"]["rank_by_vorp"]["cv_precision_at_k"]
    cv_word = "clearly beats" if dcv >= 0.03 else ("roughly ties" if dcv > -0.03 else "is WORSE than")
    if dh <= 0:
        hold = f"**does NOT beat rank by VORP on the holdout** ({dh:+d} players)"
    elif dh <= max(3, 0.05 * slots) or wins - losses <= 2:
        hold = f"beats it only narrowly on the holdout ({dh:+d} players; per season {wins} wins, {losses} losses, {ties} ties), within season-to-season noise"
    else:
        hold = f"beats it on the holdout by {dh:+d} players (per season {wins} wins, {losses} losses, {ties} ties)"
    out["vorp"] = (f"In CV ({n_cv} seasons, {cv_rng}) the chosen model {cv_word} rank-by-VORP "
                   f"(precision@k {dcv:+.3f}); it {hold}.")
    fam = cv[cv["scorer"].isin(list(m["best_by_model"]))]["cv_pr_auc"]
    out["families"] = (f"CV PR-AUC of the three model families spans {fam.min():.3f}-{fam.max():.3f}"
                       + (" (indistinguishable)." if fam.max() - fam.min() < 0.01 else "."))
    da = m["ablation_no_voter_history_cv"]["cv_pr_auc"] - m["best_by_model"][chosen]["cv_pr_auc"]
    dha = int(mk.loc[f"{chosen}_no_voter_history", "hits"] - mk.loc[chosen, "hits"])
    out["ablation"] = ("**The voter-history features did not help measurably**" if abs(da) < 0.005 and dha >= -1 else
                       ("**The voter-history features helped**" if da < 0 else "**Removing the voter-history features improved CV**"))
    out["ablation"] += f" (removing them: CV PR-AUC {da:+.3f}, holdout hits {dha:+d})."
    pdlt = (posi["p@k_2020_2023_positional"] - posi["p@k_2020_2023"])
    out["positional"] = ("**helps every scorer**" if (pdlt > 0).all() else "**hurts every scorer**" if (pdlt < 0).all()
                         else f"**mixed, no reliable gain** (better for {int((pdlt > 0).sum())}, worse for {int((pdlt < 0).sum())}, "
                              f"equal for {int((pdlt == 0).sum())} scorers)")
    tp = preds[(preds.made_all_nba == 1) & preds.picked_best_masked]
    fp = preds[(preds.made_all_nba == 0) & preds.picked_best_masked]
    mi = preds[(preds.made_all_nba == 1) & ~preds.picked_best_masked]
    fp_low = ", ".join(f"{r.player} {r.season} ({r.g} g)" for r in fp.nsmallest(3, "g").itertuples())
    mi_low = ", ".join(f"{r.player} {r.season} (VORP {r.vorp})" for r in mi.nsmallest(3, "vorp").itertuples())
    out["errors"] = (f"False positives played a median of {fp.g.median():.0f} games vs {tp.g.median():.0f} for correct picks "
                     f"(fewest: {fp_low}). This *suggests* recent voters penalise absences more than the 1980-{C.TRAIN_LAST} "
                     f"training data implies, but it has not been tested. Misses had a median VORP of {mi.vorp.median():.1f} "
                     f"vs {tp.vorp.median():.1f} for correct picks (lowest: {mi_low}). These are players whose case rests on "
                     "things the box score and team record do not capture.")
    return out


def write(reports=C.REPORTS_DIR, info: dict | None = None):
    charts(reports)
    M = json.loads((reports / "metrics.json").read_text())
    m = M["all_nba"]
    chosen = m["chosen_model"]
    overall = pd.read_csv(reports / "test_overall.csv")
    cols = ["scorer", "pr_auc", "roc_auc", "brier", "precision_at_k", "hits", "slots", "p@k_2020_2023", "p@k_2024plus"]
    masked = overall[overall["variant"] == "eligibility_mask"][cols]
    raw = overall[overall["variant"] == "raw"][["scorer", "pr_auc", "precision_at_k", "hits"]]
    posi = overall[overall["variant"] == "mask+positional"][["scorer", "p@k_2020_2023"]].rename(
        columns={"p@k_2020_2023": "p@k_2020_2023_positional"})
    posi = posi.merge(masked[["scorer", "p@k_2020_2023"]], on="scorer")
    cv = pd.DataFrame([{"scorer": n, "cv_pr_auc": b["cv_pr_auc"], "cv_precision_at_k": b["cv_precision_at_k"],
                        "params": json.dumps(b["params"])} for n, b in m["best_by_model"].items()]
                      + [{"scorer": n, **b, "params": ""} for n, b in m["baselines_cv"].items()]
                      + [{"scorer": f"{chosen}_no_voter_history", "cv_pr_auc": m["ablation_no_voter_history_cv"]["cv_pr_auc"],
                          "cv_precision_at_k": m["ablation_no_voter_history_cv"]["cv_precision_at_k"], "params": "ablation"}])
    ps = pd.read_csv(reports / "test_per_season.csv")
    ps = ps[ps["variant"] == "eligibility_mask"].pivot(index="season", columns="scorer", values="hits").reset_index()
    vs = M["all_nba_vote_share"]
    asr = pd.DataFrame(M["all_star"]["holdout"])
    mvp = pd.DataFrame([{k: v for k, v in r.items() if k != "per_season"} for r in M["mvp"]["holdout"]])
    mvp_ps = pd.DataFrame(M["mvp"]["holdout"][0]["per_season"])
    preds = pd.read_csv(reports / "test_predictions.csv")
    misses = preds[(preds.made_all_nba == 1) & (~preds.picked_best_masked)][["season", "player", "g", "vorp", "rank_best_masked", "all_nba_team"]]
    fps = preds[(preds.made_all_nba == 0) & (preds.picked_best_masked)][["season", "player", "g", "vorp", "rank_best_masked", "all_nba_share"]]
    dl = (C.DOWNLOAD_INFO.read_text() if C.DOWNLOAD_INFO.exists() else "n/a").strip()
    best_cv = m["best_by_model"][chosen]
    base_cv = m["baselines_cv"]["rank_by_vorp"]
    mk = masked.set_index("scorer")
    v = verdicts(m, chosen, mk, ps, posi, preds, cv)

    L = [f"# All-NBA / award prediction: results\n",
         "Real data: Kaggle `sumitrodatta/nba-aba-baa-stats` (Basketball-Reference export).\n",
         "```\n" + dl + "\n```\n",
         f"- Train: {m['train'][0]}-{m['train'][1]}. Holdout: {m['test'][0]}-{m['test'][1]} (the labelled seasons after training).",
         "- Tuning: forward-chaining CV inside the training range, with blocks "
         + ", ".join(f"{a}-{b}" for a, b in C.CV_BLOCKS)
         + ". Each block is scored by a model trained only on earlier seasons.",
         f"- Model and hyperparameter selection is by mean CV PR-AUC. The holdout was not used for those choices. Chosen: **{chosen}** "
         f"`{json.dumps(best_cv['params'])}`. Platt calibration (fitted on the out-of-fold CV predictions) was added *after* "
         "the first holdout run showed the class-balanced probabilities were inflated. It is monotone, so the ranking metrics "
         "(PR-AUC, ROC-AUC, precision@k) are unaffected, but the holdout Brier score and reliability table come after that inspection. "
         f"The CV evidence supports the change on its own: the uncalibrated CV Brier of the chosen configuration was "
         f"{cv_brier(chosen, best_cv['params'], reports):.4f}, versus "
         + ", ".join(f"{n} {cv_brier(n, b['params'], reports):.4f}" for n, b in m["best_by_model"].items() if n != chosen) + ".",
         "- **precision@k**: per season, the share of the real All-NBA players found in the model's top-k. k = 10 for 1980-88 "
         "and 15 from 1989.\n",
         "## Headline\n",
         f"- On the holdout with the 65-game mask, {chosen} recovers **{int(mk.loc[chosen, 'hits'])}/{int(mk.loc[chosen, 'slots'])}** "
         f"real All-NBA players. **Rank by VORP** recovers **{int(mk.loc['rank_by_vorp', 'hits'])}/{int(mk.loc['rank_by_vorp', 'slots'])}**. "
         f"PR-AUC is {mk.loc[chosen, 'pr_auc']:.3f} vs {mk.loc['rank_by_vorp', 'pr_auc']:.3f}.",
         f"- CV: precision@k {best_cv['cv_precision_at_k']:.3f} vs {base_cv['cv_precision_at_k']:.3f} for VORP; "
         f"PR-AUC {best_cv['cv_pr_auc']:.3f} vs {base_cv['cv_pr_auc']:.3f}.",
         f"- **Verdict:** {v['vorp']} {v['families']}",
         f"- Sample size: {v['sizes']}. Numbers for that era are very noisy.\n",
         "## Cross-validation (training range only)\n", _fmt(cv), "\n",
         "## Holdout, 65-game rule applied from 2024 (approximation: total games >= 65)\n", _fmt(masked), "\n",
         "Without the mask (raw model output):\n", _fmt(raw), "\n",
         "Masking ineligible players changes hits by "
         + ", ".join(f"{s} {int(mk.loc[s, 'hits']) - int(raw.set_index('scorer').loc[s, 'hits']):+d}" for s in mk.index)
         + ". All of the change is in 2024-2025, the only seasons the rule applies to.\n",
         "### Per-season hits (masked)\n", _fmt(ps), "\n",
         "![per season](figures/holdout_per_season.png)\n",
         "### Position-constrained picks for pre-2024 seasons (2 G / 2 F / 1 C per team)\n",
         "B-Ref `pos` is mapped to G/F/C (PG/SG to G, SF/PF to F, C to C). This is not the ballot position voters used.\n",
         _fmt(posi), "\n",
         f"Result (2020-2023): {v['positional']}. Not used by default.\n",
         "### Voter-history ablation\n",
         f"Removing prior All-NBA, previous-season All-NBA, prior All-Star and prior MVP top-5 features: CV PR-AUC "
         f"{m['ablation_no_voter_history_cv']['cv_pr_auc']:.3f} vs {best_cv['cv_pr_auc']:.3f} with them. Holdout hits "
         f"{int(mk.loc[f'{chosen}_no_voter_history', 'hits'])} vs {int(mk.loc[chosen, 'hits'])}. {v['ablation']} "
         "The features stay in the main model, and this result is reported either way.\n",
         "### Calibration (chosen model, holdout)\n",
         f"Before calibration the class-balanced logistic model's holdout Brier was {m['brier_uncalibrated_holdout'][chosen]:.4f}. "
         f"After Platt calibration it is {mk.loc[chosen, 'brier']:.4f}. Sum of probabilities per season (target about 15): "
         f"{m['sum_of_probabilities_per_test_season']}.\n",
         _fmt(pd.read_csv(reports / "calibration.csv")), "\n", "![calibration](figures/calibration.png)\n",
         f"Team assignment (ranks 1-5 = 1st team, ...) among correctly picked players: {m['team_assignment_accuracy_best_masked']:.0%} on the right team.\n",
         "### Errors on the holdout\n",
         "Missed (real selections ranked outside the top 15):\n", _fmt(misses), "\n",
         "False positives (picked, not selected):\n", _fmt(fps), "\n",
         f"Pattern: {v['errors']}\n",
         "## Secondary tasks\n",
         "### All-NBA vote share (HistGradientBoostingRegressor)\n",
         f"- MAE over all rows {vs['mae_all_rows']:.3f}; over vote-getters {vs['mae_vote_getters']:.3f}.",
         f"- Within-season Spearman with the real share (vote-getters) {vs['spearman_within_season_vote_getters']:.2f}.",
         f"- Used as a selector (top-k, masked): precision@k {vs['selection_precision_at_k_masked']:.3f}. "
         f"Team assignment {vs['team_assignment_accuracy_masked']:.0%}.\n",
         f"### All-Star selection (holdout {C.TEST_FIRST}-{M['all_star']['test_last']}, chosen by CV: {M['all_star']['chosen_model']})\n",
         "k = the real number of selections per season, including injury replacements. All-Stars are chosen per conference and "
         "at mid-season, and the model sees full-season stats without conference, so this is a rough fit.\n",
         _fmt(asr), "\n",
         f"### MVP (vote-share regression, holdout {C.TEST_FIRST}-{M['mvp']['test_last']})\n", _fmt(mvp), "\n",
         "Per season (model, masked):\n", _fmt(mvp_ps), "\n",
         f"With only {len(mvp_ps)} seasons, MVP winner accuracy cannot reliably separate the model from the simple metrics.\n",
         "## Explainability\n", "See [explain.md](explain.md) and `figures/shap_*.png`.\n"]
    if info:
        L += ["## Build log facts\n",
              f"- All-NBA selections removed by the population filter: {len(info.get('all_nba_dropped_by_filter', []))}; "
              f"All-Star selections removed: {info.get('all_star_dropped_by_filter')}.",
              f"- Feature NaNs in the qualified table: {info.get('feature_nan_counts')} (only the newest season's "
              "`prev_all_nba`, whose previous-season labels are unpublished; the gradient-boosting model handles NaN natively, "
              "the others impute the median).",
              f"- Label freshness (max season): {info.get('label_max_season')}.\n"]
    (reports / "report.md").write_text("\n".join(L) + "\n")
    print(f"[report] wrote {reports / 'report.md'}")


if __name__ == "__main__":
    write()
