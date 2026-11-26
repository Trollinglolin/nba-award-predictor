"""Fake nba_api endpoint classes built from the SYNTHETIC fixture (fictional players). Used only by tests_smoke.py.

The fakes return DataFrames with the column names of the real endpoints (checked against nba_api 1.12.0's
`expected_data` and the documented Advanced/clutch outputs). They never touch the network.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

TEAM_IDS = {t: 1610612700 + i for i, t in enumerate(["AAA", "BBB", "CCC", "DDD", "EEE", "FFF"])}


def _end(season_str: str) -> int:
    return int(season_str[:4]) + 1


class FakeWorld:
    """Builds per-season fake responses from the fixture's Advanced.csv / Team Summaries.csv."""

    def __init__(self, fixture_dir: Path, seed: int = 3):
        self.adv = pd.read_csv(Path(fixture_dir) / "Advanced.csv")
        self.ts = pd.read_csv(Path(fixture_dir) / "Team Summaries.csv").dropna(subset=["abbreviation"])
        self.rng = np.random.default_rng(seed)
        self.cache: dict = {}

    def season(self, s: int):
        if s in self.cache:
            return self.cache[s]
        rng = self.rng
        ts = self.ts[self.ts.season == s]
        start = pd.Timestamp(f"{s - 1}-10-20")
        tg = []
        for r in ts.itertuples():
            wins = set(rng.choice(82, int(r.w), replace=False))
            for i in range(82):
                tg.append({"SEASON_ID": f"2{s - 1}", "TEAM_ID": TEAM_IDS[r.abbreviation], "TEAM_ABBREVIATION": r.abbreviation,
                           "TEAM_NAME": r.team, "GAME_ID": f"{s}{r.abbreviation}{i:03d}",
                           "GAME_DATE": (start + pd.Timedelta(days=2 * i)).strftime("%Y-%m-%d"),
                           "MATCHUP": f"{r.abbreviation} vs. XXX", "WL": "W" if i in wins else "L"})
        tg = pd.DataFrame(tg)
        a = self.adv[(self.adv.season == s)]
        stints = a[a.team != "2TM"].copy()
        stints["order"] = stints.groupby("player_id").cumcount()
        stints["n"] = stints.groupby("player_id")["team"].transform("size")
        pg = []
        for r in stints.itertuples():
            games = tg[tg.TEAM_ABBREVIATION == r.team]
            if r.n > 1:  # traded: first stint early season, second stint late season
                half = len(games) // 2
                games = games.iloc[:half] if r.order == 0 else games.iloc[half:]
            pick = games.sample(min(int(r.g), len(games)), random_state=int(rng.integers(1e9))).sort_values("GAME_DATE")
            mpg = r.mp / r.g
            pid = 1000 + int(r.player_id[-3:])
            for gm in pick.itertuples():
                m = max(1.0, rng.normal(mpg, 4))
                pg.append({"SEASON_ID": gm.SEASON_ID, "PLAYER_ID": pid, "PLAYER_NAME": r.player, "TEAM_ID": gm.TEAM_ID,
                           "TEAM_ABBREVIATION": gm.TEAM_ABBREVIATION, "TEAM_NAME": gm.TEAM_NAME, "GAME_ID": gm.GAME_ID,
                           "GAME_DATE": gm.GAME_DATE, "MATCHUP": gm.MATCHUP, "WL": gm.WL, "MIN": round(m),
                           "PTS": int(max(0, rng.normal(r.per, 8))), "REB": int(rng.integers(0, 15)),
                           "AST": int(rng.integers(0, 12)), "STL": int(rng.integers(0, 4)), "BLK": int(rng.integers(0, 4))})
        pg = pd.DataFrame(pg)
        players = a.drop_duplicates("player_id")
        ids = 1000 + players.player_id.str[-3:].astype(int)
        adv = pd.DataFrame({"PLAYER_ID": ids, "PLAYER_NAME": players.player, "PIE": 0.1 + 0.01 * (players.per - 15).values,
                            "NET_RATING": (players.bpm * 2).values})
        clutch = pd.DataFrame({"GROUP_SET": "Overall", "PLAYER_ID": ids, "PLAYER_NAME": players.player,
                               "PTS": (players.per * 3).values, "PLUS_MINUS": (players.bpm * 4).values})
        est = pd.DataFrame({"PLAYER_ID": ids, "PLAYER_NAME": players.player, "E_NET_RATING": (players.bpm * 1.5).values})
        stand = pd.DataFrame({"LeagueID": "00", "SeasonID": f"2{s - 1}", "TeamID": [TEAM_IDS[t] for t in ts.abbreviation],
                              "Conference": "East", "PlayoffRank": ts.w.rank(ascending=False, method="first").astype(int).values,
                              "WINS": ts.w.values, "LOSSES": ts.l.values})
        self.cache[s] = {"gamelog_P": pg, "gamelog_T": tg, "standings": stand, "dash_adv": adv, "clutch": clutch,
                         "est_metrics": est}
        return self.cache[s]


def make_endpoint_factory(world: FakeWorld, fail_once: set | None = None, calls: list | None = None):
    """Returns a replacement for nba_fetch._endpoint_class. fail_once: {(endpoint, season_end)} that time out once."""
    fail_once = set(fail_once or ())
    calls = calls if calls is not None else []

    def factory(name):
        class Fake:
            def __init__(self, season, timeout=None, proxy=None, **kw):
                s = _end(season)
                calls.append((name, s))
                if (name, s) in fail_once:
                    fail_once.discard((name, s))
                    raise TimeoutError("simulated timeout")
                self.df = world.season(s)[name] if s in set(world.adv.season) else pd.DataFrame(columns=["PLAYER_ID"])

            def get_data_frames(self):
                return [self.df]
        return Fake
    return factory
