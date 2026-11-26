"""Central configuration: paths, season splits, thresholds, expected raw schema.

Season convention (from the Kaggle / Basketball-Reference export): `season` is the
season's ENDING year, so 2025 means 2024-25.
"""
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
KAGGLE_DIR = RAW_DIR / "kaggle_bref"
KAGGLE_ZIP = RAW_DIR / "kaggle_bref.zip"
DOWNLOAD_INFO = RAW_DIR / "DOWNLOAD_INFO.txt"
PROCESSED_DIR = DATA_DIR / "processed"
TABLE_PATH = PROCESSED_DIR / "player_seasons.csv"
MODELS_DIR = ROOT / "models"
REPORTS_DIR = ROOT / "reports"
FIG_DIR = REPORTS_DIR / "figures"

KAGGLE_URL = "https://www.kaggle.com/api/v1/datasets/download/sumitrodatta/nba-aba-baa-stats"

# ---------------------------------------------------------------- seasons / splits
FIRST_SEASON = 1980            # 3-point era; bpm/vorp complete from here
TRAIN_LAST = 2019              # train on FIRST_SEASON..TRAIN_LAST
TEST_FIRST = 2020              # chronological holdout: TEST_FIRST..(last labelled season)
# Forward-chaining CV blocks used for tuning (all inside the training range).
CV_BLOCKS = [(2000, 2004), (2005, 2009), (2010, 2014), (2015, 2019)]
# Rule changes verified in the data (end-year seasons):
POSITIONLESS_FROM = 2024       # 2023-24: positionless All-NBA ballot (2024 3rd team has no C)
GAMES_RULE_FROM = 2024         # 2023-24: 65-game eligibility rule (all 2024+ selections have g >= 65)
GAMES_RULE_MIN_G = 65          # approximation: real rule counts games with 20+ min and has a 62-game injury exception
ALLSTAR_SKIP = [1999]          # lockout season, no All-Star game

# ---------------------------------------------------------------- population filter
MIN_G = 40
MIN_MP = 1000
# For an in-progress season the thresholds are scaled by (team games played / 82).
FULL_SEASON_G = 82

# ---------------------------------------------------------------- features
# Metrics turned into within-season z-scores (among the qualified population).
Z_COLS = ["per", "ws", "ws_48", "bpm", "obpm", "dbpm", "vorp", "ts_percent", "usg_percent",
          "pts_per_game", "trb_per_game", "ast_per_game", "stl_per_game", "blk_per_game",
          "mp_per_game", "ows", "dws"]
# Metrics also turned into within-season ranks (1 = best).
RANK_COLS = ["vorp", "ws", "bpm", "per", "pts_per_game"]
TEAM_COLS = ["win_pct", "srs", "n_rtg"]

FEATURES = (
    [f"z_{c}" for c in Z_COLS]
    + [f"rank_{c}" for c in RANK_COLS]
    + ["g_share", "age", "team_win_pct", "team_srs", "team_n_rtg", "team_win_rank", "team_playoffs",
       "traded", "prior_all_nba", "prev_all_nba", "prior_all_star", "prior_mvp_top5"]
)

BASE_FEATURES = list(FEATURES)

# ---------------------------------------------------------------- v2 feature groups
# Kaggle-derived (available offline). Every "prior"/"prev" feature uses only seasons strictly before the row's season.
FEATURE_GROUPS = {
    # MVP voter fatigue / repeat-winner effects
    "fatigue": ["prior_mvp_wins", "mvp_last_season", "mvp_streak", "seasons_since_mvp", "prev_mvp_share"],
    # improvement vs last season (season-relative z deltas) + comeback from a short season
    "jump": ["d_z_vorp", "d_z_bpm", "d_z_ws_48", "d_z_pts_per_game", "prev_g_share"],
    # teammates' health and quality
    "teammates": ["z_teammates_missed_value", "teammate_star_games_missed", "z_best_teammate_vorp_pace"],
    # narrative proxies (no media data exists here; these are measurable stand-ins)
    "narrative": ["all_star_this_season", "team_win_improvement", "team_best_share", "team_vorp_share",
                  "stat_leader"],
    # Basketball-Reference on/off (team success with the player on vs off the floor), 1997+
    "onoff": ["z_onoff_net", "z_on_court_pm"],
}
KAGGLE_FROM = {"onoff": 1997}
# nba_api-derived (need the separate fetch; see nba_fetch.py). All available from 1996-97 (season 1997).
NBA_GROUPS = {
    "with_without": ["ww_win_diff_shrunk", "win_pct_with"],
    "availability": ["g_counted_share"],
    "seed": ["conf_seed", "top2_seed"],
    "nba_adv": ["z_pie", "z_net_rating"],
    "clutch": ["z_clutch_pts", "z_clutch_pm"],
    "est": ["z_e_net_rating"],
    "highlights": ["triple_doubles_rate", "games_40pts_rate"],
}

# ---------------------------------------------------------------- v3 (reports/improvement_plan_2024_25.md)
OPTIONAL_FILES = ["Player Shooting.csv"]          # B-Ref shooting splits, 1997+
# Archetypes: frozen K-Means model from the sibling project, applied to 1997+ by its apply_frozen.py
ARCHETYPE_CSV = ROOT.parent / "nba-archetype-pipeline" / "outputs" / "kaggle" / "player_archetypes_1997plus.csv"
ARCHETYPE_FROM = 1997
ARCHETYPES = ["Interior Scorer", "Scoring Creator", "Pass-First Guard", "Rim-Running Big",
              "Movement / Spot-up Shooter", "Low-Usage Role Wing"]
FINISHER_BIG_STATS = ["z_trb_per_game", "z_blk_per_game", "z_dws", "z_ts_percent"]
V3_TEAM_BLOCK_DROP = ["team_srs", "team_n_rtg", "team_win_rank"]          # H1a
V3_GROUPS = {
    "H3_role": ["team_pts_share", "top2_on_contender"],
    "H4_seed": ["conf_seed", "top2_seed"],
    # Low-Usage Role Wing is the reference archetype (left out so the one-hots are not collinear)
    "H6a_archetype": ["arch_interior_scorer", "arch_scoring_creator", "arch_pass_first_guard", "arch_rim_running_big",
                      "arch_movement_spot_up_shooter", "finisher_big"]
                     + [f"fb_x_{c}" for c in FINISHER_BIG_STATS] + ["self_created_2p", "dunk_share"],
    "H7_defense": ["def_pos", "def_neg", "offensive_ace", "def_neg_x_ace"],
    "H8_improver": ["team_scoring_leader", "lead_improver"],
}

# ---------------------------------------------------------------- nba_api
NBA_API_DIR = RAW_DIR / "nba_api"
NBA_FIRST_SEASON = 1997          # 1996-97: first season with advanced/clutch dashboards and play-by-play era data
V2_TRAIN_FIRST = 1997            # experiment B (needs 1997+ features)
V2_CV_BLOCKS = [(2005, 2009), (2010, 2014), (2015, 2019)]   # shared by every v2 experiment so CV numbers compare
WW_SHRINK_GAMES = 10             # with/without win% difference shrunk by n_without / (n_without + 10)
MIN_GROUP_COVERAGE = 0.8         # an nba_api group is used only if >= 80% of training rows have it
# B-Ref team codes -> NBA abbreviations (only used to break name ties in the crosswalk)
BREF_TO_NBA_TEAM = {"BRK": "BKN", "CHO": "CHA", "PHO": "PHX", "NOK": "NOK", "NOH": "NOH", "CHH": "CHH",
                    "NJN": "NJN", "SEA": "SEA", "VAN": "VAN", "WSB": "WAS", "UTA": "UTA", "GSW": "GSW", "SAS": "SAS",
                    "NOP": "NOP", "NYK": "NYK"}
# Manual crosswalk overrides: normalised B-Ref name -> normalised NBA name (nicknames / spelling differences).
# Each verified against the nba_api game logs (same season, team and games played) from the 2026-11-12 fetch.
NAME_ALIASES: dict = {
    "ronald murray": "flip murray",        # 2003-2010
    "eugene jeter": "pooh jeter",          # 2011
    "jeenathan williams": "nate williams", # 2023-2025
    "nigel hayes": "nigel hayes davis",    # 2018
    "wang zhizhi": "wang zhi zhi",         # 2001-2005
    "ron holland": "ronald holland",       # 2025-26 onward (NBA lists "Ronald Holland II")
    # nba_api game logs list these family-name-first players with the family name twice
    "cui yongxi": "cui cui",               # 2025 BKN
    "sun yue": "sun sun",                  # 2009 LAL
    "ha seung jin": "ha ha",               # 2005-2006 POR
}

# ---------------------------------------------------------------- tasks
PRIMARY_TASK = "made_all_nba"

# ---------------------------------------------------------------- expected raw schema
# (verified against the 2026-11-01 Kaggle copy; tests_smoke checks the real files against this)
EXPECTED_COLUMNS = {
    "Advanced.csv": ["season", "lg", "player", "player_id", "age", "team", "pos", "g", "gs", "mp", "per",
                     "ts_percent", "x3p_ar", "f_tr", "orb_percent", "drb_percent", "trb_percent", "ast_percent",
                     "stl_percent", "blk_percent", "tov_percent", "usg_percent", "ows", "dws", "ws", "ws_48",
                     "obpm", "dbpm", "bpm", "vorp"],
    "Player Per Game.csv": ["season", "lg", "player", "player_id", "age", "team", "pos", "g", "gs", "mp_per_game",
                            "fg_per_game", "fga_per_game", "fg_percent", "x3p_per_game", "x3pa_per_game",
                            "x3p_percent", "x2p_per_game", "x2pa_per_game", "x2p_percent", "e_fg_percent",
                            "ft_per_game", "fta_per_game", "ft_percent", "orb_per_game", "drb_per_game",
                            "trb_per_game", "ast_per_game", "stl_per_game", "blk_per_game", "tov_per_game",
                            "pf_per_game", "pts_per_game"],
    "Team Summaries.csv": ["season", "lg", "team", "abbreviation", "playoffs", "age", "w", "l", "pw", "pl", "mov",
                           "sos", "srs", "o_rtg", "d_rtg", "n_rtg", "pace"],
    "End of Season Teams.csv": ["season", "lg", "type", "number_tm", "player", "player_id", "position"],
    "End of Season Teams (Voting).csv": ["season", "lg", "type", "number_tm", "position", "player", "player_id",
                                         "age", "pts_won", "pts_max", "share", "x1st_tm", "x2nd_tm", "x3rd_tm"],
    "Player Award Shares.csv": ["season", "award", "player", "player_id", "age", "first", "pts_won", "pts_max",
                                "share", "winner"],
    "All-Star Selections.csv": ["player", "player_id", "team", "season", "lg", "replaced"],
    "Player Play By Play.csv": ["season", "lg", "player", "player_id", "team", "g", "mp",
                                "on_court_plus_minus_per_100_poss", "net_plus_minus_per_100_poss"],
}
LABEL_FILES = ["End of Season Teams.csv", "End of Season Teams (Voting).csv", "Player Award Shares.csv",
               "All-Star Selections.csv"]

RANDOM_STATE = 42
