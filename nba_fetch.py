"""Cache-first, resumable fetch of the nba_api extras (needs network access to stats.nba.com).

    .venv/bin/python -u nba_fetch.py                         # all endpoints, 1996-97 .. 2026-27 (186 calls)
    .venv/bin/python -u nba_fetch.py --first 2015-16         # fewer seasons
    .venv/bin/python -u nba_fetch.py --endpoints gamelog_P gamelog_T
    .venv/bin/python -u nba_fetch.py --proxy http://host:port
    .venv/bin/python -u nba_fetch.py --refresh-current       # re-fetch the in-progress season only

One raw file per request: data/raw/nba_api/<endpoint>/<season>.csv.gz (untouched API output). Existing files are skipped,
so rerunning the same command fetches only what is missing. A request that keeps failing is skipped and listed at the end.
An empty (0-row) answer is cached too and reported, so it is not re-requested forever (delete the file to retry).
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import pandas as pd

import config as C

# name -> (module, class, extra kwargs, season-parameter name)
ENDPOINTS = {
    "gamelog_P": ("leaguegamelog", "LeagueGameLog", {"player_or_team_abbreviation": "P"}, "season"),
    "gamelog_T": ("leaguegamelog", "LeagueGameLog", {"player_or_team_abbreviation": "T"}, "season"),
    "standings": ("leaguestandingsv3", "LeagueStandingsV3", {}, "season"),
    "dash_adv": ("leaguedashplayerstats", "LeagueDashPlayerStats",
                 {"measure_type_detailed_defense": "Advanced", "per_mode_detailed": "PerGame"}, "season"),
    "clutch": ("leaguedashplayerclutch", "LeagueDashPlayerClutch", {"per_mode_detailed": "Totals"}, "season"),
    "est_metrics": ("playerestimatedmetrics", "PlayerEstimatedMetrics", {}, "season"),
}


def season_str(end_year: int) -> str:
    return f"{end_year - 1}-{str(end_year)[2:]}"


def season_end(s: str) -> int:
    return int(s[:4]) + 1


def _endpoint_class(name):
    import importlib
    mod, cls, _, _ = ENDPOINTS[name]
    return getattr(importlib.import_module(f"nba_api.stats.endpoints.{mod}"), cls)


def fetch(endpoints, seasons, cache_dir=C.NBA_API_DIR, pause=0.8, timeout=60, retries=5, proxy=None,
          refresh=(), sleep=time.sleep, log=print):
    """Returns {"calls": n_http_calls, "fetched": [...], "skipped": [...], "failed": [...], "empty": [...]}."""
    cache_dir = Path(cache_dir)
    todo = [(e, s) for e in endpoints for s in seasons
            if (e, s) in refresh or not (cache_dir / e / f"{season_str(s)}.csv.gz").exists()]
    n_cached = len(endpoints) * len(seasons) - len(todo)
    log(f"[nba_fetch] {len(todo)} requests to make ({n_cached} already cached). "
        f"Expect roughly {len(todo) * 3 / 60:.0f}-{len(todo) * 8 / 60:.0f} min (player game logs are the slow ones).")
    out = {"calls": 0, "fetched": [], "skipped": n_cached, "failed": [], "empty": []}
    for i, (e, s) in enumerate(todo, 1):
        cls = _endpoint_class(e)
        _, _, kwargs, season_param = ENDPOINTS[e]
        path = cache_dir / e / f"{season_str(s)}.csv.gz"
        path.parent.mkdir(parents=True, exist_ok=True)
        for attempt in range(1, retries + 1):
            out["calls"] += 1
            t0 = time.time()
            try:
                df = cls(**{season_param: season_str(s)}, **kwargs, timeout=timeout, proxy=proxy).get_data_frames()[0]
                tmp = path.with_name(path.name + ".part")
                df.to_csv(tmp, index=False, compression="gzip")
                tmp.rename(path)
                tag = "EMPTY" if df.empty else f"{len(df):,} rows"
                log(f"[nba_fetch] {i}/{len(todo)} {e} {season_str(s)}: {tag} ({time.time() - t0:.1f}s)")
                (out["empty"] if df.empty else out["fetched"]).append(f"{e} {season_str(s)}")
                break
            except Exception as exc:  # timeouts, connection resets, rate limits, malformed JSON
                wait = 5 * 2 ** (attempt - 1)
                log(f"[nba_fetch] {i}/{len(todo)} {e} {season_str(s)}: {type(exc).__name__} "
                    f"(try {attempt}/{retries}{'' if attempt == retries else f', waiting {wait}s'})")
                if attempt < retries:
                    sleep(wait)
        else:
            out["failed"].append(f"{e} {season_str(s)}")
        sleep(pause)
    log(f"[nba_fetch] done: {len(out['fetched'])} fetched, {out['skipped']} cached, {len(out['empty'])} empty, "
        f"{len(out['failed'])} failed, {out['calls']} HTTP calls")
    if out["empty"]:
        log("[nba_fetch] empty answers (cached; delete the file to retry): " + ", ".join(out["empty"]))
    if out["failed"]:
        log("[nba_fetch] FAILED (rerun the same command to retry only these): " + ", ".join(out["failed"]))
    return out


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--first", default=season_str(C.NBA_FIRST_SEASON), help="first season, e.g. 1996-97")
    ap.add_argument("--last", default=None, help="last season, e.g. 2026-27 (default: newest season in the Kaggle data)")
    ap.add_argument("--endpoints", nargs="+", default=list(ENDPOINTS), choices=list(ENDPOINTS))
    ap.add_argument("--proxy", default=None)
    ap.add_argument("--timeout", type=int, default=60)
    ap.add_argument("--pause", type=float, default=0.8)
    ap.add_argument("--refresh-current", action="store_true", help="re-fetch the last season (in progress)")
    args = ap.parse_args(argv)
    if args.last is None:
        adv = C.KAGGLE_DIR / "Advanced.csv"
        last = int(pd.read_csv(adv, usecols=["season"])["season"].max()) if adv.exists() else 2027
    else:
        last = season_end(args.last)
    seasons = list(range(season_end(args.first), last + 1))
    refresh = {(e, last) for e in args.endpoints} if args.refresh_current else set()
    print(f"[nba_fetch] endpoints {args.endpoints}; seasons {season_str(seasons[0])}..{season_str(seasons[-1])} "
          f"({len(seasons)}); cache {C.NBA_API_DIR}", flush=True)
    return fetch(args.endpoints, seasons, pause=args.pause, timeout=args.timeout, proxy=args.proxy, refresh=refresh,
                 log=lambda m: print(m, flush=True))


if __name__ == "__main__":
    main()
