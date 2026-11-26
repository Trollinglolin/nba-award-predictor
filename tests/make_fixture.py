"""Write a tiny SYNTHETIC fixture with the real Kaggle column layout (fictional players/teams, made-up numbers).

    .venv/bin/python tests/make_fixture.py      # (re)writes tests/fixture/*.csv

Used only by tests_smoke.py. Nothing in it is real NBA data.
Layout: seasons 2012-2025, 6 teams, 7 players per team, 2 traded players per season (2TM + per-team rows),
All-NBA / voting / MVP labels up to 2024 (2025 = unlabelled "inference" season), All-Star labels up to 2025.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config as C  # noqa: E402

OUT = Path(__file__).resolve().parent / "fixture"
TEAMS = ["AAA", "BBB", "CCC", "DDD", "EEE", "FFF"]
POS = ["PG", "SG", "SF", "PF", "C"]
SEASONS = range(2012, 2026)
LABEL_LAST, ALLSTAR_LAST = 2024, 2025


def make(out: Path = OUT, seed: int = 7) -> Path:
    rng = np.random.default_rng(seed)
    out.mkdir(parents=True, exist_ok=True)
    n_players = 70
    base_q = rng.normal(0, 1, n_players)
    adv, pg, ts, an, vote, mvp, ast = [], [], [], [], [], [], []
    for s in SEASONS:
        roster = rng.choice(n_players, size=len(TEAMS) * 7, replace=False)
        team_w = rng.integers(20, 62, len(TEAMS))
        for t, w in zip(TEAMS, team_w):
            ts.append({"season": s, "lg": "NBA", "team": f"Fictional {t}", "abbreviation": t, "playoffs": bool(w > 41),
                       "age": 27.0, "w": float(w), "l": float(82 - w), "pw": float(w), "pl": float(82 - w),
                       "mov": (w - 41) / 3, "sos": 0.0, "srs": (w - 41) / 3, "o_rtg": 112.0, "d_rtg": 112 - (w - 41) / 3,
                       "n_rtg": (w - 41) / 3, "pace": 99.0})
        ts.append({"season": s, "lg": "NBA", "team": "League Average", "abbreviation": np.nan, "playoffs": False})
        season_rows = []
        for j, pid in enumerate(roster):
            team = TEAMS[j // 7]
            q = base_q[pid] + rng.normal(0, 0.4)
            g = int(rng.integers(30, 83))
            mp = float(g * rng.uniform(15, 37))
            row = {"season": s, "lg": "NBA", "player": f"Test Player {pid:03d}", "player_id": f"testpl{pid:03d}",
                   "age": float(22 + pid % 12), "team": team, "pos": POS[pid % 5], "g": g, "gs": float(g // 2), "mp": mp,
                   "q": q}
            season_rows.append(row)
        # two traded players per season: split into 2TM + two stints
        rows_out = []
        for j, row in enumerate(season_rows):
            if j in (0, 8):
                other = TEAMS[(TEAMS.index(row["team"]) + 1) % len(TEAMS)]
                g1 = row["g"] // 2
                tm = dict(row, team="2TM")
                st1 = dict(row, g=g1, mp=row["mp"] * g1 / row["g"])
                st2 = dict(row, team=other, g=row["g"] - g1, mp=row["mp"] * (row["g"] - g1) / row["g"])
                rows_out += [tm, st1, st2]
            else:
                rows_out.append(row)
        for r in rows_out:
            q, frac = r["q"], r["g"] / 82
            a = {k: r[k] for k in ["season", "lg", "player", "player_id", "age", "team", "pos", "g", "gs", "mp"]}
            a.update(per=15 + 4 * q, ts_percent=0.56 + 0.02 * q, x3p_ar=0.35, f_tr=0.25, orb_percent=5.0,
                     drb_percent=15.0, trb_percent=10.0, ast_percent=15 + 5 * q, stl_percent=1.5, blk_percent=1.5,
                     tov_percent=12.0, usg_percent=20 + 4 * q, ows=frac * (3 + 2 * q), dws=frac * (2 + q),
                     ws=frac * (5 + 3 * q), ws_48=0.1 + 0.05 * q, obpm=1.2 * q, dbpm=0.8 * q, bpm=2 * q,
                     vorp=frac * (1 + 1.5 * q))
            adv.append(a)
            p = {k: r[k] for k in ["season", "lg", "player", "player_id", "age", "team", "pos", "g", "gs"]}
            p.update({c: 1.0 for c in C.EXPECTED_COLUMNS["Player Per Game.csv"] if c not in p})
            p.update(mp_per_game=r["mp"] / r["g"], pts_per_game=max(2, 13 + 6 * q), trb_per_game=5 + q,
                     ast_per_game=3 + q, stl_per_game=1.0, blk_per_game=0.6, tov_per_game=2.0)
            pg.append(p)
        combined = pd.DataFrame(season_rows)  # one row per player-season
        combined["score"] = combined["q"] + combined["g"] / 82 + rng.normal(0, 0.3, len(combined))
        order = combined.sort_values("score", ascending=False).reset_index(drop=True)
        if s <= LABEL_LAST:
            for i, r in order.head(20).iterrows():
                tm = ["1st", "2nd", "3rd"][i // 5] if i < 15 else "ORV"
                if i < 15:
                    an.append({"season": s, "lg": "NBA", "type": "All-NBA", "number_tm": tm, "player": r["player"],
                               "player_id": r["player_id"], "position": "G"})
                vote.append({"season": s, "lg": "nba", "type": "all_nba", "number_tm": tm, "position": "G",
                             "player": r["player"], "player_id": r["player_id"], "age": int(r["age"]),
                             "pts_won": 100.0 - 5 * i, "pts_max": 100.0, "share": (100 - 5 * i) / 100,
                             "x1st_tm": 0.0, "x2nd_tm": 0.0, "x3rd_tm": 0.0})
            for i, r in order.head(6).iterrows():
                mvp.append({"season": s, "award": "nba mvp", "player": r["player"], "player_id": r["player_id"],
                            "age": int(r["age"]), "first": 0.0, "pts_won": 100.0 - 15 * i, "pts_max": 100.0,
                            "share": (100 - 15 * i) / 100, "winner": i == 0})
        if s <= ALLSTAR_LAST:
            for _, r in order.head(24).iterrows():
                ast.append({"player": r["player"], "player_id": r["player_id"], "team": "East", "season": s,
                            "lg": "NBA", "replaced": False})
    pbp = [{**{k: a[k] for k in ["season", "lg", "player", "player_id", "team", "g", "mp"]},
            "on_court_plus_minus_per_100_poss": 3 * (a["per"] - 15) / 4 + rng.normal(0, 2),
            "net_plus_minus_per_100_poss": 2 * (a["per"] - 15) / 4 + rng.normal(0, 3)} for a in adv]
    files = {"Advanced.csv": adv, "Player Per Game.csv": pg, "Team Summaries.csv": ts, "End of Season Teams.csv": an,
             "End of Season Teams (Voting).csv": vote, "Player Award Shares.csv": mvp, "All-Star Selections.csv": ast,
             "Player Play By Play.csv": pbp}
    for fname, rows in files.items():
        df = pd.DataFrame(rows)
        df = df[C.EXPECTED_COLUMNS[fname]]
        df.to_csv(out / fname, index=False)
    (out / "README.txt").write_text("SYNTHETIC fixture: fictional players and teams, made-up numbers. Not NBA data.\n")
    return out


if __name__ == "__main__":
    print(make())
