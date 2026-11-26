"""Smoke tests: no network, runs in well under a minute.

    .venv/bin/python tests_smoke.py

Real-data checks run only if data/raw/kaggle_bref/ exists. The end-to-end run uses the SYNTHETIC fixture in
tests/fixture/ (fictional players) and writes only to a temporary directory.
"""
from __future__ import annotations

import json
import sys
import tempfile
import traceback
from pathlib import Path

import numpy as np
import pandas as pd

import config as C
import evaluate as E
import features as F

FIXTURE = Path(__file__).resolve().parent / "tests" / "fixture"
NO_CACHE = Path(tempfile.gettempdir()) / "nba_award_tests_no_nba_cache"  # never the real data/raw/nba_api
RESULTS = []
sys.path.insert(0, str(Path(__file__).resolve().parent / "tests"))


def test(fn):
    def run():
        try:
            fn()
            RESULTS.append((fn.__name__, "ok", ""))
        except Exception as exc:  # noqa: BLE001
            RESULTS.append((fn.__name__, "FAIL", f"{type(exc).__name__}: {exc}\n{traceback.format_exc(limit=3)}"))
    run.__name__ = fn.__name__
    return run


def fixture_table(nba_cache=NO_CACHE, archetype_csv=None):
    return F.build_table(F.load_raw(FIXTURE), first_season=2012, log=lambda *_: None, nba_cache=nba_cache,
                         reports_dir=None, archetype_csv=archetype_csv)


# ------------------------------------------------------------------ schema
@test
def test_fixture_schema_matches_expected():
    for fname, cols in C.EXPECTED_COLUMNS.items():
        got = list(pd.read_csv(FIXTURE / fname, nrows=1).columns)
        assert got == cols, f"{fname}: fixture columns drifted from config.EXPECTED_COLUMNS"


@test
def test_real_schema_if_present():
    if not (C.KAGGLE_DIR / "Advanced.csv").exists():
        return
    for fname, cols in C.EXPECTED_COLUMNS.items():
        got = set(pd.read_csv(C.KAGGLE_DIR / fname, nrows=1).columns)
        missing = [c for c in cols if c not in got]
        assert not missing, f"{fname}: missing {missing}"


@test
def test_real_labels_and_spot_check_if_present():
    if not (C.KAGGLE_DIR / "Advanced.csv").exists():
        return
    raw = F.load_raw()
    L = F.label_tables(raw)
    n = L["all_nba"][L["all_nba"]["season"] >= C.FIRST_SEASON].groupby("season").size()
    assert (n[n.index <= 1988] == 10).all(), "1980-1988 should have 10 All-NBA players (two teams)"
    assert (n[n.index >= 1989] == 15).all(), "1989+ should have 15 All-NBA players"
    adv = raw["Advanced.csv"]
    j = adv[(adv.player_id == "jokicni01") & (adv.season == 2024)].iloc[0]
    assert (j.per, j.ws_48, j.bpm, j.vorp) == (31.0, 0.299, 13.2, 10.6), "Jokic 2024 spot check failed"


# ------------------------------------------------------------------ table construction
@test
def test_one_row_per_player_season_and_2tm():
    t, _ = fixture_table()
    assert not t.duplicated(["player_id", "season"]).any()
    raw = F.load_raw(FIXTURE)
    adv = raw["Advanced.csv"]
    tms = adv[adv.team == "2TM"].merge(t[["player_id", "season"]], on=["player_id", "season"])
    assert len(tms) > 0, "fixture should contain qualified traded players"
    tm = tms.iloc[0]
    row = t[(t.player_id == tm.player_id) & (t.season == tm.season)].iloc[0]
    assert row.team == "2TM" and row.traded == 1 and row.g == tm.g, "combined 2TM row must supply the stats"
    stints = adv[(adv.player_id == tm.player_id) & (adv.season == tm.season) & (adv.team != "2TM")]
    teams = F.team_table(raw["Team Summaries.csv"]).set_index(["season", "abbreviation"])
    wp = [teams.loc[(tm.season, s.team), "win_pct"] for s in stints.itertuples()]
    expect = np.average(wp, weights=stints.mp)
    assert abs(row.team_win_pct - expect) < 1e-9, "team win% must be minutes-weighted over stints"


@test
def test_games_sane_and_filter_logged():
    t, info = fixture_table()
    assert (t.g <= 90).all() and (t.mp > 0).all()
    assert (t.loc[t.traded == 0, "g"] <= t.loc[t.traded == 0, "team_g"]).all(), "non-traded g must be <= team games"
    assert "all_nba_dropped_by_filter" in info


@test
def test_labels_nan_when_unpublished():
    t, info = fixture_table()
    last = info["label_max_season"]["all_nba"]
    assert t.loc[t.season > last, "made_all_nba"].isna().all()
    assert t.loc[t.season <= last, "made_all_nba"].notna().all()


@test
def test_voter_history_strictly_prior():
    raw = F.load_raw(FIXTURE)
    kw = dict(first_season=2012, log=lambda *_: None, nba_cache=NO_CACHE, reports_dir=None)
    t1, _ = F.build_table(raw, **kw)
    # erase season-2020 labels: features of 2020 rows must not change, 2021 rows must
    raw2 = dict(raw)
    for f in ["End of Season Teams.csv", "All-Star Selections.csv", "Player Award Shares.csv"]:
        raw2[f] = raw[f][raw[f].season != 2020]
    t2, _ = F.build_table(raw2, **kw)
    cols = ["prior_all_nba", "prev_all_nba", "prior_all_star", "prior_mvp_top5", "prior_mvp_wins", "mvp_last_season",
            "mvp_streak", "seasons_since_mvp", "prev_mvp_share"]
    a = t1[t1.season == 2020].set_index("player_id")[cols]
    b = t2[t2.season == 2020].set_index("player_id")[cols]
    pd.testing.assert_frame_equal(a.sort_index(), b.sort_index())
    a21 = t1[t1.season == 2021]["prior_all_nba"].sum()
    b21 = t2[t2.season == 2021]["prior_all_nba"].sum()
    assert a21 > b21, "2021 prior counts should include 2020 selections"


@test
def test_season_split_and_cv_blocks():
    import train as T
    t, _ = fixture_table()
    old = (C.FIRST_SEASON, C.TRAIN_LAST, C.TEST_FIRST)
    C.FIRST_SEASON, C.TRAIN_LAST, C.TEST_FIRST = 2012, 2020, 2021
    try:
        tr, te = T.split(t, "made_all_nba", 2024)
        assert tr.season.max() < te.season.min() and not set(tr.season) & set(te.season)
    finally:
        C.FIRST_SEASON, C.TRAIN_LAST, C.TEST_FIRST = old
    for lo, hi in C.CV_BLOCKS:  # real config: all CV blocks inside the training range
        assert C.FIRST_SEASON < lo <= hi <= C.TRAIN_LAST
    assert C.TRAIN_LAST < C.TEST_FIRST


@test
def test_mvp_fatigue_values_and_label_exclusion():
    t, _ = fixture_table()
    raw = F.load_raw(FIXTURE)
    aw = raw["Player Award Shares.csv"]
    win = aw[aw.winner].set_index("season")["player_id"]
    s = 2020
    r = t[(t.player_id == win[s - 1]) & (t.season == s)]
    if len(r):
        r = r.iloc[0]
        expect_streak = 1
        while s - 1 - expect_streak in win.index and win[s - 1 - expect_streak] == win[s - 1]:
            expect_streak += 1
        assert r.mvp_last_season == 1 and r.mvp_streak == expect_streak
        assert r.prior_mvp_wins == int((win.loc[:s - 1] == win[s - 1]).sum())
    # the All-Star label must never be a feature of the All-Star model (v1 features)
    assert "all_star_this_season" not in C.FEATURES and "all_star" not in C.FEATURES


@test
def test_jump_and_teammates():
    t, _ = fixture_table()
    a = t[["player_id", "season", "z_vorp", "d_z_vorp"]]
    m = a.merge(a.assign(season=a.season + 1)[["player_id", "season", "z_vorp"]], on=["player_id", "season"],
                suffixes=("", "_prev"))
    assert np.allclose(m.d_z_vorp, m.z_vorp - m.z_vorp_prev), "jump = z this season - z last season"
    assert t.loc[t.season == t.season.min(), "d_z_vorp"].isna().all(), "no previous season -> NaN"
    st = pd.DataFrame({"season": 2020, "team": "AAA", "player_id": ["a", "b", "c"], "g": [82, 60, 70],
                       "mp": [2800.0, 2100.0, 1400.0], "bpm": [5.0, 4.0, -1.0], "vorp": [5.0, 3.0, 0.2],
                       "team_g": 82.0, "n_rows": 1})
    out = F.teammate_features(st).set_index("player_id")
    exp_b = 4.0 * (2100 / 60) * 22 / (48 * 82)  # b missed 22 games at 35 mpg with BPM 4
    assert abs(out.loc["a", "teammates_missed_value"] - exp_b) < 1e-9
    assert out.loc["a", "teammate_star_games_missed"] == 22 and out.loc["b", "teammate_star_games_missed"] == 0
    assert out.loc["a", "team_best_share"] == 1 and out.loc["b", "team_best_share"] == 0


# ------------------------------------------------------------------ nba_api layer
@test
def test_nba_fetch_cache_resume_timeout():
    import fake_nba
    import nba_fetch
    world = fake_nba.FakeWorld(FIXTURE)
    calls = []
    orig = nba_fetch._endpoint_class
    nba_fetch._endpoint_class = fake_nba.make_endpoint_factory(world, fail_once={("clutch", 2015)}, calls=calls)
    try:
        with tempfile.TemporaryDirectory() as d:
            seasons = [2014, 2015, 2030]  # 2030 does not exist -> empty answer, cached and reported
            r1 = nba_fetch.fetch(list(nba_fetch.ENDPOINTS), seasons, cache_dir=d, sleep=lambda *_: None, log=lambda *_: None)
            n_req = len(nba_fetch.ENDPOINTS) * len(seasons)
            assert r1["calls"] == n_req + 1, "one simulated timeout -> exactly one retry"
            assert not r1["failed"] and len(r1["empty"]) == len(nba_fetch.ENDPOINTS)
            r2 = nba_fetch.fetch(list(nba_fetch.ENDPOINTS), seasons, cache_dir=d, sleep=lambda *_: None, log=lambda *_: None)
            assert r2["calls"] == 0, "second run must be served entirely from the cache"
            # persistent failure: fails 5 times -> skipped and listed, not fatal
            always = fake_nba.make_endpoint_factory(world, calls=calls)
            class Boom:
                def __init__(self, *a, **k):
                    raise ConnectionError("down")
            nba_fetch._endpoint_class = lambda name: Boom if name == "standings" else always(name)
            r3 = nba_fetch.fetch(["standings", "est_metrics"], [2016], cache_dir=d, sleep=lambda *_: None, log=lambda *_: None)
            assert r3["failed"] == ["standings 2015-16"] and r3["fetched"] == ["est_metrics 2015-16"]
    finally:
        nba_fetch._endpoint_class = orig


@test
def test_gamelog_with_without_and_counted_games():
    import nba_features as NF
    dates = pd.date_range("2024-10-22", periods=10, freq="2D")
    gt = pd.DataFrame({"season": 2025, "TEAM_ID": 1, "GAME_ID": [f"g{i}" for i in range(10)], "GAME_DATE": dates,
                       "WL": list("WWWWWWLLLL")})
    played = [0, 1, 2, 3, 6, 7, 8]  # 4 W + 3 L with him; without: games 4,5 (W), 9 (L)
    gp = pd.DataFrame({"season": 2025, "PLAYER_ID": 7, "PLAYER_NAME": "Fake Guy", "TEAM_ID": 1, "TEAM_ABBREVIATION": "AAA",
                       "GAME_ID": [f"g{i}" for i in played], "GAME_DATE": dates[played],
                       "MIN": [25, 25, "24:30", 30, 18, 16, 15], "PTS": [40, 10, 10, 10, 10, 10, 10],
                       "REB": [10, 1, 1, 1, 1, 1, 1], "AST": [10, 1, 1, 1, 1, 1, 1], "STL": 0, "BLK": 0})
    r = NF.gamelog_features(gp, gt, k=10).iloc[0]
    assert (r.g20, r.g15, r.g_counted) == (4, 3, 6), "20+ min games, plus at most two 15-20 min games"
    assert abs(r.win_pct_with - 4 / 7) < 1e-9 and abs(r.win_pct_without - 2 / 3) < 1e-9
    assert abs(r.ww_win_diff_shrunk - (4 / 7 - 2 / 3) * 3 / 13) < 1e-9
    assert r.triple_doubles == 1 and r.games_40pts_rate == 0.1
    assert r.team_games_after_last == 1, "team games after his last appearance (injury-exception signal)"


@test
def test_crosswalk_names():
    import nba_features as NF
    assert NF.norm_name("Nikola Jokić") == NF.norm_name("Nikola Jokic") == "nikola jokic"
    assert NF.norm_name("Ronald Holland II") == "ronald holland" and NF.norm_name("Gary Payton Jr.") == "gary payton"
    assert NF.norm_name("D'Angelo Russell") == "dangelo russell"
    assert NF.norm_name("Ömer Aşık") == "omer asik" and NF.norm_name("Egor Dёmin") == "egor demin"  # dotless i, Cyrillic ё
    assert NF.norm_name("Tibor Pleiß") == "tibor pleiss"
    xw2, _ = NF.crosswalk(pd.DataFrame({"season": 2025, "player_id": ["cuiyo01x"], "player": ["Yongxi Cui"], "g": [5],
                                        "bref_teams": ["BRK"]}),
                          pd.DataFrame({"season": 2025, "PLAYER_ID": [9], "nba_name": ["Cui Yongxi"], "nba_games": [5],
                                        "nba_teams": ["BKN"]}))
    assert xw2.method.tolist() == ["token_order"], "family-name-first order should match"
    bref = pd.DataFrame({"season": 2020, "player_id": ["jokicni01", "smithja01", "smithja02", "robincl01", "nobody01"],
                         "player": ["Nikola Jokić", "Jason Smith", "Jason Smith", "Clifford Robinson", "Zed Nobody"],
                         "g": [73, 50, 20, 70, 5], "bref_teams": ["DEN", "BRK", "PHO", "POR", "SAS"]})
    nba = pd.DataFrame({"season": 2020, "PLAYER_ID": [1, 2, 3, 4], "nba_name": ["Nikola Jokic", "Jason Smith", "Jason Smith",
                                                                               "Cliff Robinson"],
                        "nba_games": [73, 20, 50, 69], "nba_teams": ["DEN", "PHX", "BKN", "POR"]})
    xw, un = NF.crosswalk(bref, nba)
    m = xw.set_index("player_id")["PLAYER_ID"].to_dict()
    assert m == {"jokicni01": 1, "smithja01": 3, "smithja02": 2, "robincl01": 4}, m
    assert xw.set_index("player_id").loc["robincl01", "method"] == "fuzzy" and list(un.player_id) == ["nobody01"]


# ------------------------------------------------------------------ evaluation helpers
@test
def test_eligibility_and_positional_topk():
    df = pd.DataFrame({"season": [2024] * 3 + [2020] * 3, "g": [60, 70, 80, 60, 70, 80], "season_scale": 1.0,
                       "pos_group": ["G", "G", "C", "G", "G", "G"]})
    s = pd.Series([0.9, 0.5, 0.1, 0.9, 0.5, 0.1])
    m = E.apply_eligibility(df, s)
    assert np.isneginf(m[0]) and m[3] == 0.9, "65-game mask applies from 2024 only"
    big = pd.DataFrame({"season": 2020, "pos_group": ["G"] * 10 + ["F"] * 6 + ["C"] * 4})
    sc = pd.Series(np.arange(20, 0, -1), dtype=float)
    pick = E.top_k(big, sc, {2020: 10}, positional=True)
    assert big[pick].pos_group.value_counts().to_dict() == {"G": 4, "F": 4, "C": 2}


# ------------------------------------------------------------------ end-to-end on the synthetic fixture
@test
def test_end_to_end_fixture():
    import explain as X
    import predict as P
    import train as T
    t, _ = fixture_table()
    saved = {k: getattr(C, k) for k in ["FIRST_SEASON", "TRAIN_LAST", "TEST_FIRST", "CV_BLOCKS"]}
    saved_grids = T.GRIDS
    C.FIRST_SEASON, C.TRAIN_LAST, C.TEST_FIRST, C.CV_BLOCKS = 2012, 2020, 2021, [(2017, 2018), (2019, 2020)]
    T.GRIDS = {"logreg": [{"C": 1.0}], "random_forest": [{"max_depth": 4, "min_samples_leaf": 5}],
               "hist_gb": [{"learning_rate": 0.1, "max_iter": 30, "max_leaf_nodes": 8, "min_samples_leaf": 5,
                            "l2_regularization": 0.0}]}
    try:
        with tempfile.TemporaryDirectory() as d:
            d = Path(d)
            (d / "SYNTHETIC_FIXTURE_OUTPUT.txt").write_text("synthetic fixture run, not real results\n")
            raw = F.load_raw(FIXTURE)
            L = F.label_tables(raw)
            counts = {"all_nba": L["all_nba"].groupby("season").size().to_dict(),
                      "all_star": L["all_star"].groupby("season").size().to_dict()}
            m = T.main(table=t, label_counts=counts, log=lambda *_: None, out_models=d / "models",
                       out_reports=d / "reports")
            assert m["all_nba"]["test"] == [2021, 2024]
            assert (d / "models" / "all_nba_best.joblib").exists()
            X.main(models_dir=d / "models", reports_dir=d / "reports", table=t, log=lambda *_: None)
            assert (d / "reports" / "explain.md").exists()
            season = P.latest_complete_season(t)
            assert season == 2025
            out = P.score_season(t, season, models_dir=d / "models")
            assert (out.pred_team > 0).sum() == 15 and out.p_all_nba.between(0, 1).all()
            json.loads((d / "reports" / "metrics.json").read_text())

            # ---- v2 with a fake nba_api cache built through the real fetch code
            import experiments as XP
            import fake_nba
            import nba_fetch
            orig = nba_fetch._endpoint_class
            nba_fetch._endpoint_class = fake_nba.make_endpoint_factory(fake_nba.FakeWorld(FIXTURE))
            try:
                nba_fetch.fetch(list(nba_fetch.ENDPOINTS), list(range(2012, 2026)), cache_dir=d / "nba_api",
                                sleep=lambda *_: None, log=lambda *_: None)
            finally:
                nba_fetch._endpoint_class = orig
            t2, info2 = F.build_table(raw, first_season=2012, log=lambda *_: None, nba_cache=d / "nba_api",
                                      reports_dir=d / "reports")
            xw = info2["nba_api"]["crosswalk"]
            assert xw["match_rate"] > 0.99 and xw["games_mismatch_gt2"] == 0, xw
            assert t2["g_counted"].notna().mean() > 0.99 and t2["ww_win_diff_shrunk"].notna().mean() > 0.99
            T.main(table=t2, label_counts=counts, log=lambda *_: None, out_models=d / "models", out_reports=d / "reports")
            sv = {k: getattr(C, k) for k in ["V2_TRAIN_FIRST", "V2_CV_BLOCKS"]}
            sg = (XP.LR_GRID, XP.HGB_GRID)
            C.V2_TRAIN_FIRST, C.V2_CV_BLOCKS = 2013, [(2017, 2018), (2019, 2020)]
            XP.LR_GRID, XP.HGB_GRID = [{"C": 1.0}], [T.GRIDS["hist_gb"][0]]
            try:
                r = XP.main(t2, counts, log=lambda *_: None, out_models=d / "models", out_reports=d / "reports")
                XP.write_report(r, info2, out_reports=d / "reports")
            finally:
                for k, v in sv.items():
                    setattr(C, k, v)
                XP.LR_GRID, XP.HGB_GRID = sg
            assert any("+everything" in s for s in r["all_nba"]["set"]), "nba_api groups should be usable on the fake cache"
            assert (d / "models" / "all_nba_v2.joblib").exists() and (d / "reports" / "v2_report.md").exists()
            out2 = P.score_season(t2, 2025, models_dir=d / "models")
            assert out2.attrs["exact_eligibility"] and (out2.pred_team > 0).sum() == 15
    finally:
        for k, v in saved.items():
            setattr(C, k, v)
        T.GRIDS = saved_grids


# ------------------------------------------------------------------ v3 features
@test
def test_v3_role_features():
    st = pd.DataFrame({"season": 2020, "team": "AAA", "player_id": ["a", "b", "c", "d"], "g": [82, 60, 70, 10],
                       "mp": [2800.0, 2100.0, 1400.0, 100.0], "bpm": [5.0, 4.0, -1.0, 0.0], "vorp": [5.0, 3.0, 0.2, 0.0],
                       "team_g": 82.0, "n_rows": 1, "pts_tot": [2000.0, 1500.0, 600.0, 50.0],
                       "usg_percent": [28.0, 31.0, 15.0, 40.0]})
    out = F.teammate_features(st).set_index("player_id")
    assert abs(out["team_pts_share"].sum() - 1) < 1e-9 and out["team_pts_share"].le(1).all()
    assert list(out["team_pts_rank"]) == [1, 2, 3, 4]
    assert out.loc["b", "team_usg_rank"] == 1 and out.loc["a", "team_usg_rank"] == 2
    assert np.isnan(out.loc["d", "team_usg_rank"]), "non-rotation player (100 min, 40% usage) must not be ranked"


def _v3_rows():
    return pd.DataFrame({
        "season": [2025, 2025, 2025, 2025, 2020],
        "player": ["finisher", "engine big", "guard rim-runner", "creator", "finisher 2020"],
        "pos": ["C", "C", "PG", "SG", "C"],
        "archetype": ["Rim-Running Big", "Rim-Running Big", "Rim-Running Big", "Scoring Creator", "Rim-Running Big"],
        "team_pts_rank": [2.0, 1.0, 2.0, 1.0, 3.0], "team_usg_rank": [4.0, 1.0, 3.0, 1.0, 5.0],
        "team_win_rank": [6.0, 3.0, 20.0, 2.0, 1.0], "team_win_improvement": [0.1, 0.2, 0.0, 0.15, np.nan],
        "z_pts_per_game": [0.5, 2.0, 0.0, 1.8, 0.2], "z_dws": [2.0, 1.0, 0.5, -0.5, 1.5],
        "z_dbpm": [1.5, 0.5, 0.2, -1.0, 1.0], "z_stl_per_game": [0.0, 0.1, 1.0, -0.2, 0.0],
        "z_blk_per_game": [2.5, 1.0, 0.0, -0.4, 2.0], "z_trb_per_game": [2.0, 1.5, 0.5, -0.5, 2.2],
        "z_ts_percent": [1.5, 0.8, -0.2, 0.3, 1.1]})


@test
def test_v3_derived_features_and_rule():
    q = F.v3_features(_v3_rows()).set_index("player")
    assert np.allclose(q["def_pos"] + q["def_neg"], q["def_z"])
    assert q["finisher_big"].to_dict() == {"finisher": 1, "engine big": 0, "guard rim-runner": 0, "creator": 0,
                                           "finisher 2020": 1}, "finisher = RRB at PF/C and not the team's top scorer"
    assert q.loc["creator", "offensive_ace"] == 1 and q.loc["creator", "def_neg_x_ace"] < 0
    assert abs(q.loc["creator", "lead_improver"] - 0.15) < 1e-12 and q.loc["finisher", "lead_improver"] == 0
    assert q.loc["finisher 2020", "lead_improver"] == 0, "missing improvement counts as none"
    assert np.allclose(q["z_pts_per_game_sq"], q["z_pts_per_game"] ** 2)
    r = F.finisher_big_rule(q.reset_index()).set_index("player")
    for c in C.FINISHER_BIG_STATS:
        assert r.loc["finisher", c] == min(q.loc["finisher", c], 0), "2025 finisher capped at league average"
        assert r.loc["finisher 2020", c] == q.loc["finisher 2020", c], "positional-era seasons untouched"
        assert r.loc["engine big", c] == q.loc["engine big", c]


@test
def test_v3_no_label_inputs_and_archetype_join():
    rows = _v3_rows()
    labelled = rows.assign(made_all_nba=1.0, all_star=1.0, mvp_share=0.9, all_nba_share=0.8)
    a = F.v3_features(rows)
    b = F.v3_features(labelled)[a.columns]
    pd.testing.assert_frame_equal(a, b)  # same-season labels never feed a v3 feature
    t0, _ = fixture_table()
    with tempfile.TemporaryDirectory() as d:
        s = t0[["player_id", "season"]].drop_duplicates().head(20)
        csv = Path(d) / "arch.csv"
        s.rename(columns={"season": "season_end"}).assign(archetype="Scoring Creator").to_csv(csv, index=False)
        t, _ = fixture_table(archetype_csv=csv)
    m = t.merge(s, on=["player_id", "season"])
    assert len(m) == len(s) and (m["archetype"] == "Scoring Creator").all() and (m["arch_scoring_creator"] == 1).all()
    assert t0["archetype"].isna().all() and t0["finisher_big"].isna().all(), "no archetype file -> NaN, not 0"


if __name__ == "__main__":
    import warnings
    warnings.filterwarnings("ignore")
    for name, obj in list(globals().items()):
        if name.startswith("test_") and callable(obj):
            obj()
    for name, status, msg in RESULTS:
        print(f"{status:4s} {name}" + (f"\n{msg}" if msg else ""))
    n_fail = sum(s != "ok" for _, s, _ in RESULTS)
    print(f"{len(RESULTS) - n_fail}/{len(RESULTS)} passed")
    sys.exit(1 if n_fail else 0)
