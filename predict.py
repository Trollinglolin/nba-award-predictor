"""Score a season with the saved models and print the predicted All-NBA teams.

Usage:
    .venv/bin/python predict.py                 # newest COMPLETE season in the data (2026 = 2025-26 in the 2026-11-01 copy)
    .venv/bin/python predict.py --season 2024   # a labelled season: also shows the real selections
    .venv/bin/python predict.py --season 2027   # in-progress season: allowed, with a warning
    .venv/bin/python predict.py --no-eligibility --top 25

Notes:
- Models were trained on END-OF-SEASON stats. A season in progress is a different distribution (small samples,
  thresholds pro-rated by games played), so treat those numbers as a rough "if the season ended today".
- The 2024+ 65-game rule is approximated as g >= 65 (pro-rated for an in-progress season); the real rule counts
  games of 20+ minutes and has exceptions that this data cannot see.
"""
from __future__ import annotations

import argparse

import joblib
import numpy as np
import pandas as pd

import config as C
import evaluate as E
import features as F


def latest_complete_season(table: pd.DataFrame, min_scale: float = 0.9) -> int:
    s = table.groupby("season")["season_scale"].first()
    return int(s[s >= min_scale].index.max())


def score_season(table: pd.DataFrame, season: int, models_dir=C.MODELS_DIR, eligibility=True) -> pd.DataFrame:
    df = table[table["season"] == season].copy()
    if df.empty:
        raise SystemExit(f"no qualified rows for season {season}")
    path = next(f for f in (models_dir / "all_nba_v3.joblib", models_dir / "all_nba_v2.joblib",
                            models_dir / "all_nba_best.joblib") if f.exists())
    best = joblib.load(path)
    df.attrs["all_nba_model"] = f"{best.get('set', 'v1')} ({best['name']})"
    feats = best["features"]
    p = pd.Series(best["model"].predict_proba(df[feats])[:, 1], index=df.index)
    df["p_all_nba"] = p
    exact = "g_counted" in df.columns and df["g_counted"].notna().any()
    df.attrs["exact_eligibility"] = bool(exact)
    games = E.eligibility_games(df, exact)
    df["games_for_rule"] = games
    df["eligible_65"] = ~((df["season"] >= C.GAMES_RULE_FROM) & (games < C.GAMES_RULE_MIN_G * df["season_scale"]))
    if exact and "team_games_after_last" in df.columns:
        # 62-game exception needs a season-ending injury: 62-64 counted games and he missed his team's last 3+ games
        df["injury_exception_possible"] = (~df["eligible_65"] & (games >= 62) & (df["team_games_after_last"] >= 3))
    score = E.apply_eligibility(df, p, rule=eligibility, exact=exact)
    df["rank"] = score.rank(ascending=False, method="first").astype(int)
    df["pred_team"] = np.where(df["rank"] <= 15, (df["rank"] - 1) // 5 + 1, 0)
    vs = joblib.load(models_dir / "all_nba_vote_share.joblib")
    df["pred_vote_share"] = vs["model"].predict(df[vs["features"]]).clip(0, 1)
    mv2 = models_dir / "mvp_v2.joblib"
    mb = joblib.load(mv2 if mv2.exists() else models_dir / "mvp_share.joblib")
    df.attrs["mvp_model"] = mb.get("set", "v1")
    df["pred_mvp_share"] = mb["model"].predict(df[mb["features"]]).clip(0, 1)
    b = joblib.load(models_dir / "all_star.joblib")
    df["p_all_star"] = b["model"].predict_proba(df[b["features"]])[:, 1]
    cand = models_dir / "all_nba_v3_candidate.joblib"
    if cand.exists():  # opt-in v3 candidate (not CV-validated): features + finisher-big rule, scored beside v1
        cb = joblib.load(cand)
        x = F.finisher_big_rule(df) if cb.get("rule") == "finisher_big_rule" else df
        df["p_candidate"] = cb["model"].predict_proba(x[cb["features"]])[:, 1]
        sc = E.apply_eligibility(df, df["p_candidate"], rule=eligibility, exact=exact)
        df["rank_candidate"] = sc.rank(ascending=False, method="first").astype(int)
        df["pred_team_candidate"] = np.where(df["rank_candidate"] <= 15, (df["rank_candidate"] - 1) // 5 + 1, 0)
    return df.sort_values("rank")


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--season", type=int, help="season ENDING year (2026 = 2025-26)")
    ap.add_argument("--top", type=int, default=20)
    ap.add_argument("--no-eligibility", action="store_true", help="ignore the 2024+ 65-game rule")
    ap.add_argument("--rebuild", action="store_true", help="rebuild the feature table from data/raw first")
    ap.add_argument("--candidate", action="store_true",
                    help="also show the opt-in v3 candidate (user hypotheses H6c/H7/H8; not CV-validated)")
    args = ap.parse_args(argv)

    if args.rebuild or not C.TABLE_PATH.exists():
        table, _ = F.build_table(F.load_raw())
    else:
        table = pd.read_csv(C.TABLE_PATH, low_memory=False)
    season = args.season or latest_complete_season(table)
    scale = float(table.loc[table["season"] == season, "season_scale"].iloc[0])
    labelled = table.loc[table["season"] == season, "made_all_nba"].notna().any()
    print(f"Season {season - 1}-{str(season)[2:]} (season={season}); fraction of schedule played ~{scale:.0%}; "
          f"real All-NBA labels in data: {'yes' if labelled else 'no'}")
    if scale < 0.9:
        print("WARNING: season in progress. Models were trained on end-of-season stats; filters and the 65-game rule "
              "are pro-rated. Treat this as 'if the season ended today'.")
    df = score_season(table, season, eligibility=not args.no_eligibility)
    print(f"All-NBA model: {df.attrs['all_nba_model']}; MVP model: {df.attrs['mvp_model']}")
    if season >= C.GAMES_RULE_FROM:
        how = ("exact: nba_api games with 20+ min, plus up to two 15-20 min games" if df.attrs["exact_eligibility"]
               else "approximation: total games, since nba_api game logs are not cached for this season")
        print(f"65-game rule {'IGNORED' if args.no_eligibility else 'applied (' + how + ')'}")
    cols = ["rank", "player", "team", "pos", "g", "games_for_rule", "vorp", "bpm", "pts_per_game", "team_win_pct", "eligible_65",
            "p_all_nba", "pred_team", "pred_vote_share", "pred_mvp_share", "p_all_star"]
    if labelled:
        df["real_team"] = df["all_nba_team"].astype(int)
        cols.append("real_team")
    out = df[cols].head(args.top).copy()
    for c in ["p_all_nba", "pred_vote_share", "pred_mvp_share", "p_all_star", "team_win_pct"]:
        out[c] = out[c].round(3)
    print(out.to_string(index=False))
    for t in (1, 2, 3):
        print(f"Predicted All-NBA {['1st', '2nd', '3rd'][t - 1]} team: " + ", ".join(df.loc[df["pred_team"] == t, "player"]))
    if args.candidate and "p_candidate" in df:
        print("\nOpt-in v3 candidate (defense gate + improver features, finisher-big rule from 2024; NOT CV-validated, "
              "see reports/v3_candidate.md):")
        for t in (1, 2, 3):
            print(f"  Candidate {['1st', '2nd', '3rd'][t - 1]} team: "
                  + ", ".join(df.sort_values("rank_candidate").loc[lambda d: d["pred_team_candidate"] == t, "player"]))
        a, b = set(df.loc[df["pred_team"] > 0, "player"]), set(df.loc[df["pred_team_candidate"] > 0, "player"])
        print(f"  In candidate top 15 only: {', '.join(sorted(b - a)) or 'none'}; in v1 top 15 only: "
              f"{', '.join(sorted(a - b)) or 'none'}")
    pool = df if args.no_eligibility else df[df["eligible_65"]]
    mvp = pool.sort_values("pred_mvp_share", ascending=False).head(3)
    print("Predicted MVP top 3: " + ", ".join(f"{r.player} ({r.pred_mvp_share:.2f})" for r in mvp.itertuples()))
    excl = df[~df["eligible_65"] & (df["p_all_nba"] >= 0.2)].sort_values("p_all_nba", ascending=False)
    if not args.no_eligibility and len(excl):
        print(f"Excluded by the 65-game rule (counted games < {C.GAMES_RULE_MIN_G}): "
              + ", ".join(f"{r.player} (counted={int(r.games_for_rule)}, p={r.p_all_nba:.2f})" for r in excl.itertuples())
              + ". Rerun with --no-eligibility to include them.")
        if "injury_exception_possible" in df.columns and df.loc[excl.index, "injury_exception_possible"].any():
            ex = df.loc[excl.index][df.loc[excl.index, "injury_exception_possible"]]
            print("  Possible 62-game season-ending-injury exception (62-64 counted games and missed his team's last 3+ "
                  "games; the data cannot confirm an injury or the league's ruling): "
                  + ", ".join(f"{r.player} (missed last {int(r.team_games_after_last)})" for r in ex.itertuples()))
    if labelled:
        hits = int(((df["pred_team"] > 0) & (df["made_all_nba"] == 1)).sum())
        k = int(df["made_all_nba"].sum())
        print(f"Overlap with the real All-NBA selections: {hits}/{k} (note: seasons <= {C.TRAIN_LAST} were used for training)")
    C.REPORTS_DIR.mkdir(exist_ok=True)
    path = C.REPORTS_DIR / f"predictions_{season}.csv"
    df.drop(columns=[c for c in df.columns if c.startswith(("z_", "rank_"))]).to_csv(path, index=False)
    print(f"saved {path}")


if __name__ == "__main__":
    main()
