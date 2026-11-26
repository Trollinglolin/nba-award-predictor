"""Build the analytical table: one row per qualified NBA player-season, with features and labels.

Decisions (each is logged when the table is built):
- NBA rows only (`lg == "NBA"`; the voting file spells it `nba`). ABA/BAA rows and "League Average" rows are dropped.
- Traded players: Basketball-Reference gives a combined row (`team` = 2TM/3TM/...) plus one row per team.
  The combined row supplies the player stats; the per-team rows are used ONLY to compute minutes-weighted
  team features (win %, SRS, net rating, win rank, playoffs) and are then dropped.
- Population: g >= MIN_G and mp >= MIN_MP, scaled by (median team games that season / 82), so shortened
  seasons (1999, 2012, 2020, 2021) and an in-progress season use proportional thresholds.
- Season-relative features: z-scores and ranks computed within each season among the qualified players.
- Voter-history features use only labels from seasons strictly before the row's season.
- Labels are NaN (not 0) for seasons whose label file has not been published yet.
"""
from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd

import config as C
import nba_features as NF

POS_GROUP = {"PG": "G", "SG": "G", "G": "G", "SF": "F", "PF": "F", "F": "F", "C": "C"}
PG_COLS = ["mp_per_game", "pts_per_game", "trb_per_game", "ast_per_game", "stl_per_game", "blk_per_game",
           "tov_per_game", "fga_per_game"]


def _nba(df: pd.DataFrame) -> pd.DataFrame:
    return df[df["lg"].str.upper() == "NBA"].copy()


def load_raw(raw_dir: Path = C.KAGGLE_DIR) -> dict[str, pd.DataFrame]:
    raw_dir = Path(raw_dir)
    raw = {f: pd.read_csv(raw_dir / f) for f in C.EXPECTED_COLUMNS}
    for f in C.OPTIONAL_FILES:  # used if present (the test fixture does not ship them)
        if (raw_dir / f).exists():
            raw[f] = pd.read_csv(raw_dir / f)
    return raw


def team_table(ts: pd.DataFrame) -> pd.DataFrame:
    t = _nba(ts).dropna(subset=["abbreviation"])  # drops "League Average"
    t = t.assign(team_g=t["w"] + t["l"], win_pct=t["w"] / (t["w"] + t["l"]),
                 playoffs=t["playoffs"].astype(bool).astype(float))
    t["win_rank"] = t.groupby("season")["win_pct"].rank(ascending=False, method="min")
    return t[["season", "abbreviation", "team_g", "win_pct", "srs", "n_rtg", "win_rank", "playoffs"]]


def label_tables(raw: dict) -> dict[str, pd.DataFrame]:
    eos = _nba(raw["End of Season Teams.csv"])
    an = eos[eos["type"] == "All-NBA"].copy()
    an["all_nba_team"] = an["number_tm"].str[0].astype(int)
    vot = _nba(raw["End of Season Teams (Voting).csv"])
    vot = vot[vot["type"] == "all_nba"]
    aw = raw["Player Award Shares.csv"]
    mvp = aw[aw["award"] == "nba mvp"].copy()
    mvp["mvp_rank"] = mvp.groupby("season")["share"].rank(ascending=False, method="min")
    ast = _nba(raw["All-Star Selections.csv"])
    return {
        "all_nba": an[["season", "player_id", "all_nba_team"]],
        "vote": vot[["season", "player_id", "share"]].rename(columns={"share": "all_nba_share"}),
        "mvp": mvp[["season", "player_id", "share", "winner", "mvp_rank"]].rename(
            columns={"share": "mvp_share", "winner": "mvp_winner"}),
        "all_star": ast[["season", "player_id"]].drop_duplicates().assign(all_star=1),
        "max_season": {
            "all_nba": int(an["season"].max()), "vote": int(vot["season"].max()),
            "mvp": int(mvp["season"].max()), "all_star": int(ast["season"].max()),
        },
    }


def _prior_count(rows: pd.DataFrame, events: pd.DataFrame) -> pd.Series:
    """For each (player_id, season) row: number of event seasons strictly before `season`."""
    ev = events[["player_id", "season"]].drop_duplicates().rename(columns={"season": "ev_season"})
    m = rows[["player_id", "season"]].reset_index().merge(ev, on="player_id", how="left")
    m["hit"] = (m["ev_season"] < m["season"]).astype(int)
    return m.groupby("index")["hit"].sum().reindex(rows.index).fillna(0).astype(int)


def teammate_features(st: pd.DataFrame) -> pd.DataFrame:
    """Per stint (player on one team): teammates' health and quality, the player's share of team value.

    st: per-team stint rows with g, mp, bpm, vorp, team_g, n_rows (1 = not traded that season)."""
    s = st.copy()
    s["rotation"] = s["mp"] >= 500 * s["team_g"] / C.FULL_SEASON_G
    s["vorp_pace"] = s["vorp"] / s["g"].clip(lower=1) * s["team_g"]
    healthy_known = s["rotation"] & (s["n_rows"] == 1)  # traded players' absences are trades, not injuries
    missed = (s["team_g"] - s["g"]).clip(lower=0)
    s["_missed_value"] = np.where(healthy_known, s["bpm"].clip(lower=0) * (s["mp"] / s["g"].clip(lower=1))
                                  * missed / (48 * s["team_g"]), 0.0)
    s["_star_missed"] = np.where(healthy_known & (s["bpm"] >= 3), missed, 0.0)
    s["_vorp_pos"] = s["vorp"].clip(lower=0)
    s["_pts"] = s["pts_tot"].fillna(0.0) if "pts_tot" in s else 0.0
    g = s.groupby(["season", "team"])
    # v3 role features: share of the team's points, rank on the team by points (all players) and by usage
    # (rotation players only, so a 10-game call-up with 35% usage does not outrank the starters)
    s["team_pts_share"] = s["_pts"] / g["_pts"].transform("sum").replace(0, np.nan)
    s["team_pts_rank"] = g["_pts"].rank(ascending=False, method="min")
    usg = s["usg_percent"].where(s["rotation"]) if "usg_percent" in s else pd.Series(np.nan, index=s.index)
    s["team_usg_rank"] = usg.groupby([s["season"], s["team"]]).rank(ascending=False, method="min")
    s["teammates_missed_value"] = g["_missed_value"].transform("sum") - s["_missed_value"]
    s["teammate_star_games_missed"] = g["_star_missed"].transform("sum") - s["_star_missed"]
    s["team_vorp_share"] = s["_vorp_pos"] / g["_vorp_pos"].transform("sum").replace(0, np.nan)
    pace = s["vorp_pace"].where(s["rotation"])
    s["_pace_rank"] = pace.groupby([s["season"], s["team"]]).rank(ascending=False, method="first")
    top1 = pace.where(s["_pace_rank"] == 1).groupby([s["season"], s["team"]]).transform("max")
    top2 = pace.where(s["_pace_rank"] == 2).groupby([s["season"], s["team"]]).transform("max")
    s["best_teammate_vorp_pace"] = np.where(s["_pace_rank"] == 1, top2, top1)
    s["team_best_share"] = (s["vorp_pace"] >= s["best_teammate_vorp_pace"].fillna(-np.inf)).astype(float)
    return s


def v3_features(q: pd.DataFrame) -> pd.DataFrame:
    """v3 candidate features (reports/improvement_plan_2024_25.md, H1b/H3/H6/H7/H8). Needs the season-relative z
    columns. Traded players' team ranks are minutes-weighted means, so 'leader' means rank <= 1.5."""
    q = q.copy()
    # H1b: scoring curvature instead of the points rank
    q["z_pts_per_game_sq"] = q["z_pts_per_game"] ** 2
    # H3: role on a contender
    leader = q["team_pts_rank"] <= 1.5
    q["top2_on_contender"] = ((q["team_pts_rank"] <= 2.5) & (q["team_win_rank"] <= 4)).astype(float)
    # H7: weak defense should not sink an offensive ace
    q["def_z"] = q[["z_dws", "z_dbpm", "z_stl_per_game", "z_blk_per_game"]].mean(axis=1)
    q["def_pos"] = q["def_z"].clip(lower=0)
    q["def_neg"] = q["def_z"].clip(upper=0)
    q["offensive_ace"] = ((q["team_usg_rank"] <= 1.5) & (q["z_pts_per_game"] >= 1.0)).astype(float)
    q["def_neg_x_ace"] = q["def_neg"] * q["offensive_ace"]
    # H8: led a much-improved team to a top-8 record (the "Ja Morant 2022" pattern)
    q["team_scoring_leader"] = leader.astype(float)
    q["lead_improver"] = (q["team_scoring_leader"] * q["team_win_improvement"].fillna(0).clip(lower=0)
                          * (q["team_win_rank"] <= 8))
    # H6: archetypes. A "finisher big" is a Rim-Running Big listed PF/C who is NOT his team's leading scorer
    # (keeps prime Shaq / Howard, who share the rim-heavy shot profile but were the offense).
    known = q["archetype"].notna()
    for a in C.ARCHETYPES:
        q[f"arch_{_slug(a)}"] = (q["archetype"] == a).astype(float).where(known)
    big = q["pos"].astype(str).str.split("-").str[0].isin(["PF", "C"])
    q["finisher_big"] = ((q["archetype"] == "Rim-Running Big") & big & ~leader).astype(float).where(known)
    for c in C.FINISHER_BIG_STATS:
        q[f"fb_x_{c}"] = q["finisher_big"] * q[c]
    return q


def _slug(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", s.lower()).strip("_")


def finisher_big_rule(df: pd.DataFrame) -> pd.DataFrame:
    """H6c, pre-registered for positionless seasons (>= POSITIONLESS_FROM): a finisher big's rebounding, blocks,
    DWS and TS% count as no better than the average qualified player (z capped at 0). No tuned parameter."""
    out = df.copy()
    m = (out["season"] >= C.POSITIONLESS_FROM) & (out["finisher_big"] == 1)
    for c in C.FINISHER_BIG_STATS:
        out.loc[m, c] = out.loc[m, c].clip(upper=0)
    return out


def _streak(events: pd.DataFrame) -> dict:
    """{(player_id, season): length of the consecutive-season streak ending at that season}."""
    out = {}
    for pid, seasons in events.groupby("player_id")["season"]:
        for y in sorted(seasons):
            out[(pid, y)] = out.get((pid, y - 1), 0) + 1
    return out


def build_table(raw: dict, first_season: int = C.FIRST_SEASON, log=print, nba_cache=C.NBA_API_DIR,
                reports_dir=C.REPORTS_DIR, archetype_csv=C.ARCHETYPE_CSV) -> tuple[pd.DataFrame, dict]:
    info: dict = {}
    load_from = first_season - 1  # one extra season so the first season has year-over-year features
    adv = _nba(raw["Advanced.csv"])
    adv = adv[adv["season"] >= load_from]
    pg = _nba(raw["Player Per Game.csv"])
    pg = pg[pg["season"] >= load_from]
    teams = team_table(raw["Team Summaries.csv"])
    teams = teams.sort_values("season")
    teams["win_improve"] = teams["win_pct"] - teams.groupby("abbreviation")["win_pct"].shift(1).where(
        teams.groupby("abbreviation")["season"].diff() == 1)

    adv["n_rows"] = adv.groupby(["player_id", "season"])["team"].transform("size")
    is_tm = adv["team"].str.fullmatch(r"\dTM", na=False)
    season_rows = adv[(adv["n_rows"] == 1) | is_tm].copy()
    stints = adv[~is_tm].copy()
    log(f"[features] Advanced NBA rows {len(adv)} -> season rows {len(season_rows)} "
        f"(traded combined rows {int(is_tm.sum())}, per-team stint rows {len(stints)})")
    assert not season_rows.duplicated(["player_id", "season"]).any(), "duplicate player-season after 2TM handling"
    bad = season_rows[(season_rows["n_rows"] > 1) & ~season_rows["team"].str.fullmatch(r"\dTM", na=False)]
    assert bad.empty, f"multi-row player-seasons without a combined row: {len(bad)}"

    # ---- per-game stats (join on the same row identity, incl. team code)
    n0 = len(season_rows)
    season_rows = season_rows.merge(pg[["season", "player_id", "team"] + PG_COLS],
                                    on=["season", "player_id", "team"], how="left", validate="1:1")
    miss_pg = int(season_rows["pts_per_game"].isna().sum())
    log(f"[features] per-game join: {n0} rows, {miss_pg} without per-game stats")

    # ---- team features: minutes-weighted over the player's stints
    st = stints.merge(teams, left_on=["season", "team"], right_on=["season", "abbreviation"], how="left")
    miss_team = int(st["win_pct"].isna().sum())
    log(f"[features] team join: {len(st)} stint rows, {miss_team} without a team match")
    st["w_mp"] = st["mp"].fillna(0).clip(lower=1e-6)
    # per-stint total points (per-team rows of the per-game file share the stint's row identity)
    pg_st = pg[["season", "player_id", "team", "g", "pts_per_game"]].drop_duplicates(["season", "player_id", "team"])
    pg_st = pg_st.assign(pts_tot=pg_st["pts_per_game"] * pg_st["g"])[["season", "player_id", "team", "pts_tot"]]
    st = st.merge(pg_st, on=["season", "player_id", "team"], how="left", validate="m:1")
    log(f"[features] stint points join: {int(st['pts_tot'].isna().sum())} of {len(st)} stints without points")
    st = teammate_features(st)
    wcols = {"win_pct": "team_win_pct", "srs": "team_srs", "n_rtg": "team_n_rtg", "win_rank": "team_win_rank",
             "playoffs": "team_playoffs", "win_improve": "team_win_improvement",
             "teammates_missed_value": "teammates_missed_value", "teammate_star_games_missed": "teammate_star_games_missed",
             "best_teammate_vorp_pace": "best_teammate_vorp_pace", "team_best_share": "team_best_share",
             "team_vorp_share": "team_vorp_share", "team_pts_share": "team_pts_share",
             "team_pts_rank": "team_pts_rank", "team_usg_rank": "team_usg_rank"}
    tmp = st[["player_id", "season", "w_mp"] + list(wcols)].copy()
    out = tmp[["player_id", "season"]].drop_duplicates().set_index(["player_id", "season"])
    for col, name in wcols.items():  # minutes-weighted mean that ignores NaN stints
        v = tmp[col]
        w = tmp["w_mp"].where(v.notna(), 0.0)
        num = (v.fillna(0) * w).groupby([tmp["player_id"], tmp["season"]]).sum()
        den = w.groupby([tmp["player_id"], tmp["season"]]).sum()
        out[name] = (num / den.replace(0, np.nan)).reindex(out.index)
    out["team_g"] = st.groupby(["player_id", "season"])["team_g"].max()
    tf = out.reset_index()
    season_rows = season_rows.merge(tf, on=["player_id", "season"], how="left", validate="1:1")

    # ---- Basketball-Reference on/off (1997+), same row identity as Advanced (incl. 2TM rows)
    if "Player Play By Play.csv" in raw:
        pbp = _nba(raw["Player Play By Play.csv"])[["season", "player_id", "team", "on_court_plus_minus_per_100_poss",
                                                    "net_plus_minus_per_100_poss"]]
        pbp = pbp.drop_duplicates(["season", "player_id", "team"])
        season_rows = season_rows.merge(pbp.rename(columns={"on_court_plus_minus_per_100_poss": "on_court_pm",
                                                            "net_plus_minus_per_100_poss": "onoff_net"}),
                                        on=["season", "player_id", "team"], how="left", validate="1:1")
        log(f"[features] on/off join: {int(season_rows.loc[season_rows.season >= 1997, 'onoff_net'].notna().sum())} of "
            f"{int((season_rows.season >= 1997).sum())} 1997+ rows have on/off")
    else:
        season_rows["on_court_pm"] = season_rows["onoff_net"] = np.nan

    # ---- Basketball-Reference shooting (1997+): self-created 2s and dunk share, same row identity
    if "Player Shooting.csv" in raw:
        sh = _nba(raw["Player Shooting.csv"])[["season", "player_id", "team", "percent_assisted_x2p_fg",
                                               "percent_dunks_of_fga"]].drop_duplicates(["season", "player_id", "team"])
        season_rows = season_rows.merge(sh, on=["season", "player_id", "team"], how="left", validate="1:1")
        season_rows["self_created_2p"] = 1.0 - season_rows["percent_assisted_x2p_fg"]
        season_rows["dunk_share"] = season_rows["percent_dunks_of_fga"]
    else:
        season_rows["self_created_2p"] = season_rows["dunk_share"] = np.nan

    # ---- player archetypes (frozen model from ../nba-archetype-pipeline, 1997+); NaN where not available
    arch_csv = Path(archetype_csv) if archetype_csv else None
    if arch_csv is not None and arch_csv.exists():
        ar = pd.read_csv(arch_csv)[["player_id", "season_end", "archetype"]].rename(columns={"season_end": "season"})
        season_rows = season_rows.merge(ar.drop_duplicates(["player_id", "season"]), on=["player_id", "season"],
                                        how="left", validate="1:1")
        log(f"[features] archetypes from {arch_csv.name}: "
            f"{int(season_rows.loc[season_rows.season >= C.ARCHETYPE_FROM, 'archetype'].notna().sum())} of "
            f"{int((season_rows.season >= C.ARCHETYPE_FROM).sum())} {C.ARCHETYPE_FROM}+ rows labelled")
    else:
        season_rows["archetype"] = np.nan
        log(f"[features] no archetype file ({arch_csv}); archetype features are NaN")

    season_rows["traded"] = season_rows["team"].str.fullmatch(r"\dTM", na=False).astype(int)
    season_rows["g_share"] = (season_rows["g"] / season_rows["team_g"]).clip(upper=1.0)
    season_rows["pos_group"] = season_rows["pos"].astype(str).str.split("-").str[0].map(POS_GROUP)

    # ---- labels
    L = label_tables(raw)
    info["label_max_season"] = L["max_season"]
    t = season_rows
    t = t.merge(L["all_nba"], on=["season", "player_id"], how="left")
    t = t.merge(L["vote"], on=["season", "player_id"], how="left")
    t = t.merge(L["mvp"], on=["season", "player_id"], how="left")
    t = t.merge(L["all_star"], on=["season", "player_id"], how="left")
    t["made_all_nba"] = t["all_nba_team"].notna().astype(float)
    t["all_nba_team"] = t["all_nba_team"].fillna(0)
    for col in ["all_nba_share", "mvp_share", "all_star"]:
        t[col] = t[col].fillna(0.0)
    t["mvp_winner"] = t["mvp_winner"].fillna(False).astype(float)
    t["mvp_rank"] = t["mvp_rank"].fillna(99)
    unlabelled = {
        "made_all_nba": t["season"] > L["max_season"]["all_nba"],
        "all_nba_team": t["season"] > L["max_season"]["all_nba"],
        "all_nba_share": t["season"] > L["max_season"]["vote"],
        "mvp_share": t["season"] > L["max_season"]["mvp"],
        "mvp_winner": t["season"] > L["max_season"]["mvp"],
        "mvp_rank": t["season"] > L["max_season"]["mvp"],
        "all_star": (t["season"] > L["max_season"]["all_star"]) | t["season"].isin(C.ALLSTAR_SKIP),
    }
    for col, mask in unlabelled.items():
        t.loc[mask, col] = np.nan

    # ---- voter-history features (strictly prior seasons, full label history incl. pre-1980)
    mvp_top5 = L["mvp"][L["mvp"]["mvp_rank"] <= 5]
    t["prior_all_nba"] = _prior_count(t, L["all_nba"])
    prev = L["all_nba"][["player_id", "season"]].assign(season=lambda d: d["season"] + 1, prev_all_nba=1)
    t = t.merge(prev, on=["player_id", "season"], how="left")
    t["prev_all_nba"] = t["prev_all_nba"].fillna(0)
    t.loc[t["season"] - 1 > L["max_season"]["all_nba"], "prev_all_nba"] = np.nan  # previous season unknown
    t["prior_all_star"] = _prior_count(t, L["all_star"])
    t["prior_mvp_top5"] = _prior_count(t, mvp_top5)

    # ---- MVP voter fatigue (strictly prior seasons)
    mvp_max = L["max_season"]["mvp"]
    winners = L["mvp"][L["mvp"]["mvp_winner"] == True][["player_id", "season"]]  # noqa: E712
    prev_unknown = t["season"] - 1 > mvp_max
    t["prior_mvp_wins"] = _prior_count(t, winners)
    streak = _streak(winners)
    t["mvp_streak"] = [streak.get((p, s - 1), 0) for p, s in zip(t["player_id"], t["season"])]
    t["mvp_last_season"] = (t["mvp_streak"] > 0).astype(float)
    last_win = winners.groupby("player_id")["season"].apply(sorted).to_dict()
    t["seasons_since_mvp"] = [min([s - y for y in last_win.get(p, []) if y < s] or [10]) for p, s in
                              zip(t["player_id"], t["season"])]
    t["seasons_since_mvp"] = t["seasons_since_mvp"].clip(upper=10)
    prev_share = L["mvp"][["player_id", "season", "mvp_share"]].assign(season=lambda d: d["season"] + 1)
    t = t.merge(prev_share.rename(columns={"mvp_share": "prev_mvp_share"}), on=["player_id", "season"], how="left")
    t["prev_mvp_share"] = t["prev_mvp_share"].fillna(0.0)
    for c in ["mvp_last_season", "prev_mvp_share", "mvp_streak"]:
        t.loc[prev_unknown, c] = np.nan

    # ---- narrative: All-Star selection THIS season (made in February, known before end-of-season voting).
    #      Never used by the All-Star model itself (that would be the label).
    t["all_star_this_season"] = t["all_star"]
    t.loc[t["season"].isin(C.ALLSTAR_SKIP), "all_star_this_season"] = 0.0

    # ---- comeback: previous season's games share (any previous NBA row, qualified or not)
    prev_gs = t[["player_id", "season", "g_share"]].assign(season=lambda d: d["season"] + 1)
    t = t.merge(prev_gs.rename(columns={"g_share": "prev_g_share"}), on=["player_id", "season"], how="left")

    # ---- population filter (thresholds scaled by the season's length)
    season_len = t.groupby("season")["team_g"].median()
    scale = (season_len / C.FULL_SEASON_G).clip(upper=1.0)
    t["season_scale"] = t["season"].map(scale)
    t["qualified"] = (t["g"] >= C.MIN_G * t["season_scale"]) & (t["mp"] >= C.MIN_MP * t["season_scale"])
    lost = t[(~t["qualified"]) & (t["made_all_nba"] == 1) & (t["season"] >= first_season)]
    info["all_nba_dropped_by_filter"] = lost[["season", "player", "g", "mp"]].to_dict("records")
    lost_as = t[(~t["qualified"]) & (t["all_star"] == 1) & (t["season"] >= first_season)]
    info["all_star_dropped_by_filter"] = int(len(lost_as))
    log(f"[features] filter g>={C.MIN_G}*scale & mp>={C.MIN_MP}*scale keeps {int(t['qualified'].sum())} of {len(t)} "
        f"rows; All-NBA selections dropped: {len(lost)}; All-Star selections dropped: {len(lost_as)}")
    q = t[t["qualified"]].copy()

    # ---- nba_api extras (empty if nba_fetch.py has not been run)
    stint_rows = adv[~is_tm][["season", "player_id", "team"]]
    nf, ninfo = NF.build(t, stint_rows, cache_dir=nba_cache, log=log, reports_dir=reports_dir)
    info["nba_api"] = ninfo
    if len(nf):
        q = q.merge(nf, on=["season", "player_id"], how="left", validate="1:1")
    for c in ["g_counted", "ww_win_diff_shrunk", "win_pct_with", "g_counted_share", "conf_seed", "top2_seed",
              "triple_doubles_rate", "games_40pts_rate", "pie", "net_rating", "clutch_pts", "clutch_pm", "e_net_rating"]:
        if c not in q.columns:
            q[c] = np.nan

    # ---- season-relative features (among qualified players)
    grp = q.groupby("season")
    z_extra = ["teammates_missed_value", "best_teammate_vorp_pace", "onoff_net", "on_court_pm",
               "pie", "net_rating", "clutch_pts", "clutch_pm", "e_net_rating"]
    for col in C.Z_COLS + z_extra:
        mu, sd = grp[col].transform("mean"), grp[col].transform("std")
        q[f"z_{col}"] = ((q[col] - mu) / sd.where(sd > 1e-12)).fillna(0.0).where(q[col].notna())
    for col in C.RANK_COLS:
        q[f"rank_{col}"] = grp[col].rank(ascending=False, method="min")
    q["age"] = q["age"].astype(float)
    q["stat_leader"] = sum((grp[c].rank(ascending=False, method="min") == 1).astype(int)
                           for c in ["pts_per_game", "trb_per_game", "ast_per_game"])

    # ---- jump: change in season-relative standing vs the previous season (NaN if not qualified last season)
    jump = ["vorp", "bpm", "ws_48", "pts_per_game"]
    prev = q[["player_id", "season"] + [f"z_{c}" for c in jump]].assign(season=lambda d: d["season"] + 1)
    q = q.merge(prev, on=["player_id", "season"], how="left", suffixes=("", "_prev"))
    for c in jump:
        q[f"d_z_{c}"] = q[f"z_{c}"] - q[f"z_{c}_prev"]
    q = q.drop(columns=[f"z_{c}_prev" for c in jump])
    q = v3_features(q)
    q = q[q["season"] >= first_season]

    nan_counts = q[C.FEATURES].isna().sum()
    info["feature_nan_counts"] = {k: int(v) for k, v in nan_counts[nan_counts > 0].items()}
    log(f"[features] qualified table {q.shape}; feature NaNs: {info['feature_nan_counts'] or 'none'}")
    per_season = q.groupby("season").agg(n=("player_id", "size"), all_nba=("made_all_nba", "sum"))
    info["rows_per_season"] = per_season.reset_index().to_dict("records")
    return q.reset_index(drop=True), info


def main():
    raw = load_raw()
    table, info = build_table(raw)
    C.PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    table.to_csv(C.TABLE_PATH, index=False)
    print(f"[features] wrote {C.TABLE_PATH} ({len(table)} rows)")
    return table, info


if __name__ == "__main__":
    main()
