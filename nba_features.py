"""Normalise the nba_api cache into per player-season features and join them to the Basketball-Reference ids.

Nothing here touches the network. If data/raw/nba_api/ is missing or empty, `build()` returns an empty frame and the
rest of the pipeline runs on Kaggle features only.

Crosswalk B-Ref slug <-> NBA PLAYER_ID, per season:
  1. exact match on normalised name (lower case, accents/punctuation removed, Jr/Sr/II/III/IV dropped, NAME_ALIASES applied);
     several candidates -> the one sharing a team, then the closest games-played count;
  2. leftovers: same last name + same first initial + a shared team + |games difference| <= 2 ("fuzzy");
  3. everything else is written to reports/crosswalk_unmatched.csv. Every match is validated by games played.
"""
from __future__ import annotations

import re
import unicodedata
from pathlib import Path

import numpy as np
import pandas as pd

import config as C

SUFFIXES = {"jr", "sr", "ii", "iii", "iv", "v"}
# letters NFKD does not decompose, and Cyrillic look-alikes used in some B-Ref names (e.g. "Dёmin")
TRANSLIT = str.maketrans({"ı": "i", "ł": "l", "Ł": "L", "ø": "o", "Ø": "O", "đ": "d", "Đ": "D", "ß": "ss", "æ": "ae",
                          "œ": "oe", "þ": "th", "ð": "d", "е": "e", "ё": "e", "а": "a", "о": "o", "с": "c", "р": "p",
                          "х": "x", "у": "y", "к": "k", "м": "m", "т": "t"})


def norm_name(name: str) -> str:
    s = unicodedata.normalize("NFKD", str(name).translate(TRANSLIT))
    s = "".join(ch for ch in s if not unicodedata.combining(ch)).lower()
    s = re.sub(r"[.'`’]", "", s)
    s = re.sub(r"[^a-z0-9 ]+", " ", s)
    toks = [t for t in s.split() if t not in SUFFIXES]
    return " ".join(toks)


def _minutes(x) -> float:
    if pd.isna(x):
        return 0.0
    if isinstance(x, str) and ":" in x:
        m, s = x.split(":")[:2]
        return float(m) + float(s) / 60
    return float(x)


def load_cache(cache_dir: Path = C.NBA_API_DIR) -> dict[str, pd.DataFrame]:
    out = {}
    cache_dir = Path(cache_dir)
    for d in sorted(p for p in cache_dir.glob("*") if p.is_dir()) if cache_dir.exists() else []:
        frames = []
        for f in sorted(d.glob("*.csv.gz")):
            df = pd.read_csv(f, compression="gzip", dtype={"GAME_ID": str, "SEASON_ID": str})
            if len(df):
                frames.append(df.assign(season=int(f.name[:4]) + 1))
        if frames:
            out[d.name] = pd.concat(frames, ignore_index=True)
    return out


# --------------------------------------------------------------------------- per-season NBA features
def gamelog_features(gp: pd.DataFrame, gt: pd.DataFrame, k: float = C.WW_SHRINK_GAMES) -> pd.DataFrame:
    gp = gp.copy()
    gp["minutes"] = gp["MIN"].map(_minutes)
    gp["GAME_DATE"] = pd.to_datetime(gp["GAME_DATE"])
    stats = gp[["PTS", "REB", "AST", "STL", "BLK"]].fillna(0)
    gp["td"] = ((stats >= 10).sum(axis=1) >= 3).astype(int)
    gp["g40"] = (gp["PTS"] >= 40).astype(int)
    agg = gp.groupby(["season", "PLAYER_ID"]).agg(
        nba_name=("PLAYER_NAME", "first"), nba_games=("GAME_ID", "nunique"),
        g20=("minutes", lambda m: int((m >= 20).sum())), g15=("minutes", lambda m: int(((m >= 15) & (m < 20)).sum())),
        triple_doubles=("td", "sum"), games_40pts=("g40", "sum"),
        nba_teams=("TEAM_ABBREVIATION", lambda t: "|".join(sorted(set(t)))),
        last_game=("GAME_DATE", "max"))
    agg["g_counted"] = agg["g20"] + agg["g15"].clip(upper=2)  # 65-game rule: 20+ min games, up to two 15-20 min games
    main = (gp.groupby(["season", "PLAYER_ID", "TEAM_ID"]).size().rename("n").reset_index()
              .sort_values("n", ascending=False).drop_duplicates(["season", "PLAYER_ID"]))
    agg = agg.join(main.set_index(["season", "PLAYER_ID"])["TEAM_ID"].rename("main_team_id"))

    # with / without: team games inside each stint window
    gt = gt.copy()
    gt["GAME_DATE"] = pd.to_datetime(gt["GAME_DATE"])
    season_bounds = gt.groupby("season")["GAME_DATE"].agg(["min", "max"])
    team_games = {key: g[["GAME_ID", "GAME_DATE", "WL"]] for key, g in gt.groupby(["season", "TEAM_ID"])}
    stints = (gp.groupby(["season", "PLAYER_ID", "TEAM_ID"])
                .agg(first=("GAME_DATE", "min"), last=("GAME_DATE", "max"), games=("GAME_ID", lambda x: set(x)))
                .reset_index().sort_values(["season", "PLAYER_ID", "first"]))
    rows = []
    for (season, pid), st in stints.groupby(["season", "PLAYER_ID"], sort=False):
        ww = np.zeros(4)  # games_with, wins_with, games_without, wins_without
        n = len(st)
        after_last = np.nan
        for j, r in enumerate(st.itertuples()):
            tg = team_games.get((season, r.TEAM_ID))
            if tg is None:
                continue
            lo = season_bounds.loc[season, "min"] if j == 0 else r.first
            hi = season_bounds.loc[season, "max"] if j == n - 1 else r.last
            win = tg[(tg["GAME_DATE"] >= lo) & (tg["GAME_DATE"] <= hi)]
            played = win["GAME_ID"].isin(r.games)
            ww += [played.sum(), ((win["WL"] == "W") & played).sum(), (~played).sum(), ((win["WL"] == "W") & ~played).sum()]
            if j == n - 1:  # final team: team games played after the player's last appearance (season-ending absence)
                after_last = int((tg["GAME_DATE"] > r.last).sum())
        rows.append({"season": season, "PLAYER_ID": pid, "games_with": ww[0], "wins_with": ww[1],
                     "games_without": ww[2], "wins_without": ww[3], "team_games_after_last": after_last})
    w = pd.DataFrame(rows).set_index(["season", "PLAYER_ID"])
    agg = agg.join(w)
    agg["win_pct_with"] = agg["wins_with"] / agg["games_with"].replace(0, np.nan)
    agg["win_pct_without"] = agg["wins_without"] / agg["games_without"].replace(0, np.nan)
    diff = (agg["win_pct_with"] - agg["win_pct_without"]).fillna(0.0)
    agg["ww_win_diff_shrunk"] = diff * agg["games_without"] / (agg["games_without"] + k)
    team_total = (agg["games_with"] + agg["games_without"]).replace(0, np.nan)
    agg["g_counted_share"] = (agg["g_counted"] / team_total).clip(upper=1)
    agg["triple_doubles_rate"] = agg["triple_doubles"] / team_total
    agg["games_40pts_rate"] = agg["games_40pts"] / team_total
    agg["team_games_nba"] = team_total
    return agg.reset_index()


def dashboard_features(cache: dict) -> pd.DataFrame:
    parts = []
    spec = {"dash_adv": {"PIE": "pie", "NET_RATING": "net_rating"},
            "clutch": {"PTS": "clutch_pts", "PLUS_MINUS": "clutch_pm"},
            "est_metrics": {"E_NET_RATING": "e_net_rating"}}
    for ep, cols in spec.items():
        df = cache.get(ep)
        if df is None:
            continue
        have = {k: v for k, v in cols.items() if k in df.columns}
        missing = set(cols) - set(have)
        if missing:
            print(f"[nba_features] {ep}: columns not in the response, skipped: {sorted(missing)}")
        if not have:
            continue
        d = df.groupby(["season", "PLAYER_ID"])[list(have)].first().rename(columns=have)
        parts.append(d)
    return pd.concat(parts, axis=1).reset_index() if parts else pd.DataFrame(columns=["season", "PLAYER_ID"])


def standings_features(st: pd.DataFrame | None) -> pd.DataFrame:
    if st is None or "PlayoffRank" not in st.columns:
        return pd.DataFrame(columns=["season", "main_team_id", "conf_seed", "top2_seed"])
    s = st[["season", "TeamID", "PlayoffRank"]].rename(columns={"TeamID": "main_team_id", "PlayoffRank": "conf_seed"})
    s["conf_seed"] = pd.to_numeric(s["conf_seed"], errors="coerce")
    s["top2_seed"] = (s["conf_seed"] <= 2).astype(float)
    return s


# --------------------------------------------------------------------------- crosswalk
def crosswalk(bref: pd.DataFrame, nba: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """bref: season, player_id, player, g, bref_teams ('|'-joined stint codes). nba: season, PLAYER_ID, nba_name, nba_games, nba_teams."""
    b = bref.copy()
    b["key"] = [C.NAME_ALIASES.get(norm_name(n), norm_name(n)) for n in b["player"]]
    b["teams_nba"] = b["bref_teams"].map(lambda t: {C.BREF_TO_NBA_TEAM.get(x, x) for x in str(t).split("|")})
    n = nba.copy()
    n["key"] = n["nba_name"].map(norm_name)
    n["teams"] = n["nba_teams"].map(lambda t: set(str(t).split("|")))
    cand = b.merge(n, on=["season", "key"], how="inner", suffixes=("", "_n"))
    cand["team_overlap"] = [len(a & c) for a, c in zip(cand["teams_nba"], cand["teams"])]
    cand["g_diff"] = (cand["g"] - cand["nba_games"]).abs()
    cand = cand.sort_values(["season", "player_id", "team_overlap", "g_diff"], ascending=[True, True, False, True])
    ncand = cand.groupby(["season", "player_id"])["PLAYER_ID"].transform("size")
    cand["method"] = np.where(ncand > 1, "exact_tiebreak", "exact")
    m1 = cand.drop_duplicates(["season", "player_id"])
    # token-order pass (family-name-first names, e.g. "Cui Yongxi" vs "Yongxi Cui"), must share a team
    rest = b[~b.set_index(["season", "player_id"]).index.isin(m1.set_index(["season", "player_id"]).index)].copy()
    if len(rest):
        rest["tok"] = rest["key"].map(lambda k: " ".join(sorted(k.split())))
        n_tok = n.assign(tok=n["key"].map(lambda k: " ".join(sorted(k.split()))))
        tk = rest.merge(n_tok, on=["season", "tok"], how="inner", suffixes=("", "_n"))
        tk["team_overlap"] = [len(a & c) for a, c in zip(tk["teams_nba"], tk["teams"])]
        tk["g_diff"] = (tk["g"] - tk["nba_games"]).abs()
        tk = tk[(tk["team_overlap"] > 0) & (tk["key"] != tk["key_n"])]
        tk = tk[tk.groupby(["season", "player_id"])["PLAYER_ID"].transform("size") == 1].assign(method="token_order")
        m1 = pd.concat([m1, tk[m1.columns.intersection(tk.columns)]], ignore_index=True)
    # fuzzy pass for the rest
    left = b.merge(m1[["season", "player_id"]], how="left", indicator=True)
    left = left[left["_merge"] == "left_only"].drop(columns="_merge")
    used = set(zip(m1["season"], m1["PLAYER_ID"]))
    n2 = n[[(s, p) not in used for s, p in zip(n["season"], n["PLAYER_ID"])]].copy()
    left["last"], left["init"] = left["key"].str.split().str[-1], left["key"].str[:1]
    n2["last"], n2["init"] = n2["key"].str.split().str[-1], n2["key"].str[:1]
    fz = left.merge(n2, on=["season", "last", "init"], how="inner", suffixes=("", "_n"))
    fz["team_overlap"] = [len(a & c) for a, c in zip(fz["teams_nba"], fz["teams"])]
    fz["g_diff"] = (fz["g"] - fz["nba_games"]).abs()
    fz = fz[(fz["team_overlap"] > 0) & (fz["g_diff"] <= 2)]
    fz = fz[fz.groupby(["season", "player_id"])["PLAYER_ID"].transform("size") == 1].assign(method="fuzzy")
    cols = ["season", "player_id", "player", "PLAYER_ID", "nba_name", "method", "g", "nba_games", "g_diff"]
    xw = pd.concat([m1[cols], fz[cols]], ignore_index=True)
    xw = xw.sort_values("g_diff").drop_duplicates(["season", "PLAYER_ID"])  # one NBA id per season
    un = b.merge(xw[["season", "player_id"]], how="left", indicator=True)
    un = un[un["_merge"] == "left_only"][["season", "player_id", "player", "g", "bref_teams"]]
    return xw.reset_index(drop=True), un.reset_index(drop=True)


def build(season_rows: pd.DataFrame, stints: pd.DataFrame, cache_dir: Path = C.NBA_API_DIR, log=print,
          reports_dir: Path | None = C.REPORTS_DIR) -> tuple[pd.DataFrame, dict]:
    """Returns (features keyed by season + B-Ref player_id, info). Empty frame if there is no cache."""
    cache = load_cache(cache_dir)
    info = {"endpoints": {k: sorted(v["season"].unique().tolist()) for k, v in cache.items()}}
    if "gamelog_P" not in cache or "gamelog_T" not in cache:
        log(f"[nba_features] no nba_api game logs in {cache_dir}; nba_api features skipped")
        return pd.DataFrame(columns=["season", "player_id"]), info
    gl = gamelog_features(cache["gamelog_P"], cache["gamelog_T"])
    dash = dashboard_features(cache)
    nba = gl.merge(dash, on=["season", "PLAYER_ID"], how="left")
    nba = nba.merge(standings_features(cache.get("standings")), on=["season", "main_team_id"], how="left")

    seasons = set(nba["season"])
    teams = stints.groupby(["season", "player_id"])["team"].agg(lambda t: "|".join(sorted(set(t)))).rename("bref_teams")
    b = season_rows[season_rows["season"].isin(seasons)][["season", "player_id", "player", "g"]].merge(
        teams.reset_index(), on=["season", "player_id"], how="left")
    xw, un = crosswalk(b, nba[["season", "PLAYER_ID", "nba_name", "nba_games", "nba_teams"]])
    info["crosswalk"] = {"bref_rows": len(b), "matched": len(xw), "unmatched": len(un),
                         "match_rate": round(len(xw) / max(len(b), 1), 4),
                         "methods": xw["method"].value_counts().to_dict(),
                         "games_mismatch_gt2": int((xw["g_diff"] > 2).sum())}
    log(f"[nba_features] crosswalk: {info['crosswalk']}")
    if reports_dir is not None:
        Path(reports_dir).mkdir(parents=True, exist_ok=True)
        un.to_csv(Path(reports_dir) / "crosswalk_unmatched.csv", index=False)
        xw[xw["g_diff"] > 2].to_csv(Path(reports_dir) / "crosswalk_games_mismatch.csv", index=False)
    feats = xw[["season", "player_id", "PLAYER_ID"]].merge(nba, on=["season", "PLAYER_ID"], how="left")
    keep = ["season", "player_id", "PLAYER_ID", "g_counted", "g20", "g15", "last_game", "team_games_after_last", "ww_win_diff_shrunk",
            "win_pct_with", "win_pct_without", "games_without", "g_counted_share", "conf_seed", "top2_seed",
            "triple_doubles_rate", "games_40pts_rate", "pie", "net_rating", "clutch_pts", "clutch_pm", "e_net_rating"]
    return feats[[c for c in keep if c in feats.columns]], info
