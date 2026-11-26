# nba-award-predictor

**Live demo:** https://trollinglolin.github.io/nba-award-predictor/

Predicts **All-NBA selection** (primary), plus All-NBA vote share, All-Star selection and MVP vote share. It uses
season-relative advanced metrics (PER, WS, WS/48, BPM, VORP), box-score production, availability, team success and
voters' history with each player, and explains the predictions with SHAP.

Full results: [`reports/report.md`](reports/report.md) (v1), [`reports/v2_report.md`](reports/v2_report.md) (new feature
groups: MVP voter fatigue, teammates' health, year-over-year jumps, team with/without, narrative proxies, nba_api extras).
Explanations: [`reports/explain.md`](reports/explain.md).

## Quick start

```bash
/opt/homebrew/opt/python@3.14/bin/python3.14 -m venv .venv      # Python 3.14.7 was used
.venv/bin/pip install -r requirements.txt
.venv/bin/python -u run_pipeline.py      # ingest (cached) -> features (+nba_api cache) -> train -> explain -> v2 -> predict, ~2 min
.venv/bin/python tests_smoke.py          # 15 offline tests, ~25 s
.venv/bin/python predict.py              # newest complete season (2026 = 2025-26)
.venv/bin/python predict.py --season 2024 --top 25 # a labelled season, with the real teams alongside
.venv/bin/python predict.py --season 2027 # in-progress season: allowed, with a warning
.venv/bin/python predict.py --no-eligibility       # ignore the 65-game rule
```

`run_pipeline.py --refresh-data` re-downloads the data (about 11 MB, one request) and records the date in
`data/raw/DOWNLOAD_INFO.txt`. Everything else runs offline from the cache. A failed download never touches the existing cache.

**If the download returns a 4xx:** on 2026-11-12 the same URL returned HTTP 200 to `curl` but **404 to python-requests**.
`ingest.py` stops on a 4xx without retrying and prints this fallback:
```bash
curl -L -m 120 -o kaggle_bref.zip "https://www.kaggle.com/api/v1/datasets/download/sumitrodatta/nba-aba-baa-stats"
.venv/bin/python ingest.py --zip kaggle_bref.zip --origin "curl from Kaggle, <date>"
```

## nba_api extras

```bash
.venv/bin/pip install -r requirements.txt                       # adds nba_api
# 1) connectivity check: 1 call, a few seconds
.venv/bin/python -u nba_fetch.py --first 2024-25 --last 2024-25 --endpoints gamelog_T
# 2) full fetch: 6 endpoints x 31 seasons (1996-97..2026-27) = 186 calls (minus the one above), about 10-25 min
.venv/bin/python -u nba_fetch.py 2>&1 | tee reports/nba_fetch_log.txt
#    if it lists FAILED requests, rerun the same command: it fetches only the missing ones
# 3) rebuild + train with the nba_api features (no network), about 2-4 min
.venv/bin/python -u run_pipeline.py 2>&1 | tee reports/pipeline_log.txt
.venv/bin/python predict.py
```

The six endpoints are cached as `data/raw/nba_api/<endpoint>/<season>.csv.gz`, untouched:
- `gamelog_P`: every player game. Gives the exact 65-game count (20+ min games, plus up to two of 15-20 min), team record with vs without
  the player, triple-doubles and 40-point games.
- `gamelog_T`: every team game.
- `standings`: conference seed.
- `dash_adv`: PIE and net rating.
- `clutch`: clutch points and +/-.
- `est_metrics`: estimated net rating.

After the fetch, check `reports/v2_report.md` ("nba_api data status") and `reports/crosswalk_unmatched.csv`. A name that didn't match
can be fixed in `config.NAME_ALIASES`. Games-played mismatches for the **in-progress 2026-27 season** are expected: the Kaggle
snapshot (2026-11-01) and your fetch are from different dates. That isn't a crosswalk bug. An endpoint that answers with 0 rows for old seasons is cached and listed. A group is used only
if it covers 80% or more of the 1997-2019 training rows.

**Live status (fetch run on 2026-11-12):**
- The run made 185 calls with 0 failures and 0 empty answers. The cache is 27 MB.
- All expected columns were present in the live responses.
- Crosswalk: **100%** of 15,010 player-seasons from 1997 on matched (14,942 exact, 60 fuzzy, 8 exact with tiebreak).
  - The 22 distinct fuzzy name pairs were checked by hand (Steve/Steven Smith, Ike/Isaac Austin, ...), and each has identical games played.
  - Fixes found on live data: names containing letters NFKD doesn't decompose or Cyrillic look-alikes (Ömer Aşık, Egor Dёmin), and
    nickname aliases (`config.NAME_ALIASES`).
  - nba_api lists three family-name-first players as "Cui Cui", "Sun Sun" and "Ha Ha".
- The 337 games-played mismatches are all in 2026-27, from the snapshot-date difference.

## v2 feature groups and what they showed

| group | what it measures |
|---|---|
| fatigue | prior MVP wins, MVP last season, win streak, seasons since last MVP, last season's MVP share (strictly prior) |
| jump | change in season-relative VORP/BPM/WS48/points vs last season; last season's games share (comebacks) |
| teammates | games missed by rotation teammates weighted by their BPM x minutes; games missed by BPM >= 3 teammates; best teammate's VORP pace |
| narrative (proxies) | All-Star this season (chosen in February, before voting); team win% change; best player on the team; share of team VORP; stat-title leader |
| onoff | Basketball-Reference on/off net rating (1997+): team success with the player on vs off the floor |
| nba_api | with/without win%, exact availability, conference seed, PIE/net rating, clutch, estimated net rating, triple-doubles/40-pt games |

*Snapshot from the run with the live nba_api cache, 2026-11-12. `reports/v2_report.md` is regenerated on every run with computed verdicts.
When nothing is adopted, `predict.py` serves the exact v1 models.*

A new group is adopted only if its CV gain over the v1 features has a Bonferroni-adjusted bootstrap CI above 0.
Each group is also tested inside its own training window (`gain_vs_window_ref`), so 1997+ groups aren't penalised for
having fewer training seasons. The comparison
uses the same 15 CV seasons (2005-2019) for every set, resampled by season.
- **All-NBA:** no group passed, from either the Kaggle data or nba_api. Each nba_api group was compared within its own window, against
  the 1997+ Kaggle + on/off set without it:
  - with/without win%: +0.002 PR-AUC, CI [-0.004, 0.008]
  - PIE/net rating: 0.000
  - conference seed: -0.004
  - clutch, estimated net rating, highlights, exact availability: all within ±0.007, with CIs spanning 0

  v2 keeps the v1 features.
- **MVP:** no group passed. The real winner's rank is capped at 10 so that outliers don't dominate: 1997+ sets train their first CV block
  on 8 seasons and rank Nash's 2005-06 MVPs in the 200s. "+everything" picks 9/15 CV winners vs 8/15 for the base, which is not
  significant.
- **The 65-game rule is now exact** for every season with game logs: 20+ min games, plus up to two of 15-20 min. `predict.py` also flags
  a possible 62-game injury exception: 62-64 counted games and missed his team's last 3+ games.
- **MVP voter fatigue:** not supported by the data. Over 2005-2019, reigning MVPs received **more** vote share than their stats predict
  (+0.09). The fatigue-looking cases (Garnett 2005, LeBron 2011) are offset by reigning MVPs who won although the model didn't pick
  them (Nash 2006, LeBron 2013).
- **Narrative:** there is no media data, so these are measurable proxies only. True narrative remains the main thing the model cannot see.

## Data

- **Source:** Kaggle [`sumitrodatta/nba-aba-baa-stats`](https://www.kaggle.com/datasets/sumitrodatta/nba-aba-baa-stats),
  a Basketball-Reference export. Downloaded with curl on **2026-11-12** from the public Kaggle endpoint (no login). The files inside
  are dated **2026-11-01**. The data derives from Basketball-Reference: `data/` is git-ignored, so don't
  redistribute the raw files, and check the Kaggle licence and Sports-Reference's terms before reusing them.
- v1 uses only this export: no `nba_api`, Sportradar or scraping. The optional `nba_api` extras (v2) come from `nba_fetch.py`, run separately.
- **`season` is the ENDING year** (2025 = 2024-25).
- **Label freshness in this copy:** All-NBA, voting and MVP labels go up to **2025**, and All-Star up to **2026**. Stats go up to 2027
  (2026-27, about 7 games in). So **2026 (2025-26) is a complete season with no All-NBA labels**, and it serves as the inference season.
  Re-download later and check `DOWNLOAD_INFO.txt` before using 2026 as a test season.

### Facts checked against the data (and enforced in tests where possible)

| Fact | Consequence |
|---|---|
| 1980-1988 had **10** All-NBA players (two teams); 15 from 1989 | precision@k uses each season's real k |
| Positionless ballot from **2024** (the 2024 3rd team has no C) | results are reported separately for 2020-2023 and 2024+ |
| Every 2024+ selection has g >= 65 | the 65-game rule is applied as a mask from 2024 |
| Voting codes `1T/2T/3T` (2022-2025) are a format change, not ties | treated as 1st/2nd/3rd |
| Traded players: a `2TM`/`3TM` combined row plus per-team rows; combined `g` can reach 88 | stats come from the combined row; team features are minutes-weighted over the per-team rows |
| No All-Star game in 1999 | that season is excluded from the All-Star task |
| Jokic 2024: PER 31.0, WS/48 .299, BPM 13.2, VORP 10.6 | spot check in the tests |
| `g >= 40 & mp >= 1000` (pro-rated for short seasons) drops **0** All-NBA selections since 1980 | logged on every build |

## Method

- **Rows:** one per qualified NBA player-season, 1980+ (11,604 rows). Thresholds are scaled by season length, which
  covers the 1999 and 2012 lockouts, the 2020 bubble, the 72-game 2021 season and any in-progress season.
- **Features (34, `config.FEATURES`):**
  - within-season z-scores of PER, WS, WS/48, BPM, OBPM, DBPM, VORP, OWS, DWS, TS%, USG%, and points/rebounds/assists/steals/blocks/minutes per game
  - within-season ranks of VORP, WS, BPM, PER and points
  - games-played share, age, traded flag
  - team win %, SRS, net rating, league win rank, playoffs
  - voter history from strictly earlier seasons: career All-NBA count, All-NBA last season, All-Star count, MVP top-5 count
- **Split:** train 1980-2019. Holdout 2020-2025 (every labelled season after training), scored once.
  Tuning uses forward-chaining CV inside the training range (blocks 2000-04, 05-09, 10-14, 15-19).
- **Models:** baselines (rank by VORP / WS / points); logistic regression (`class_weight="balanced"`); random forest;
  `HistGradientBoostingClassifier`. The model is chosen by CV PR-AUC, and probabilities are Platt-calibrated on out-of-fold CV
  predictions. XGBoost/LightGBM were not used, because they need `brew install libomp` on this machine.
- **Metrics:** PR-AUC (primary), ROC-AUC, Brier score, and **precision@k** (how many of the real All-NBA players are in the top k each season).

## Results (holdout 2020-2025, 90 All-NBA slots, 65-game mask from 2024)

*This section is a snapshot from the 2026-11-01 data copy. [`reports/report.md`](reports/report.md) is regenerated on every
run, and all of its verdict sentences are computed from the numbers.*

| scorer | real All-NBA players found | PR-AUC |
|---|---|---|
| **logistic regression (chosen by CV)** | **74 / 90** | 0.915 |
| random forest | 76 / 90 | 0.921 |
| hist gradient boosting | 75 / 90 | 0.905 |
| rank by VORP | 71 / 90 | 0.883 |
| rank by WS | 58 / 90 | 0.716 |
| rank by points | 63 / 90 | 0.748 |

- **The model beats "rank by VORP", but only narrowly on the holdout.** It finds 3 more players over 6 seasons, which is within
  noise: per season it gains 2 twice, loses 1 once and ties three times. Over the 20 CV seasons (2000-2019) the gap is clearer:
  precision@k is 0.80 vs 0.69. The three model families are indistinguishable (CV PR-AUC 0.896-0.899).
- **Voter-history features did not help.** CV PR-AUC was 0.899 with and without them, and the holdout was slightly better without them.
- **Position-constrained picks** for the pre-2024 positional ballots showed **no reliable gain**, so they are not used by default.
- **Calibration:** the balanced logistic model's raw probabilities summed to about 30 per season for 15 slots. After Platt calibration
  they sum to about 14-16, and the holdout Brier score drops from 0.041 to 0.015 (both unmasked). Calibration was added after the
  first holdout run exposed the inflation. The CV Brier (0.044 vs about 0.017 for the trees) supports it independently, and it
  does not change any ranking metric.
- **Secondary tasks:**
  - vote-share regressor: within-season Spearman 0.81 with the real vote share
  - All-Star (2020-2026): 148/186 for the model vs 136 for VORP
  - MVP winner: 4/6 seasons, tied with "rank by BPM/WS/PER"; 6 seasons is too few to separate them

### 2025-26 inference (season 2026, no labels in this data copy)

`predict.py` with the 65-game mask predicts:
- 1st team: Jokic, Gilgeous-Alexander, Durant, Mitchell, Jaylen Brown
- 2nd team: Harden, Brunson, Maxey, Towns, Murray
- 3rd team: Jalen Johnson, Kawhi Leonard, Duren, Sengun, Amen Thompson

**Caveat (now with the exact count from nba_api game logs):**
- Counted games: Luka Doncic 64, Victor Wembanyama 64 (62 of 20+ min plus two of 15-20 min), Cade Cunningham 63.
- Their model probabilities are 0.93-0.99, but all three are excluded.
- Jokic (65) and Kawhi Leonard (65) qualify exactly.
- Luka is flagged as a **possible 62-game exception**: he missed Dallas's last 5 games. If the league granted it, he would replace
  the bottom of the 3rd team.
- `--no-eligibility` shows the unmasked ranking.

These are model outputs, not checked against the real 2025-26 selections.

## How to read the outputs

- `p_all_nba` is a calibrated probability. Across a season the values sum to about 15, so 0.7 means that historically about 7 in
  10 such players were selected.
- The **rank** matters more than the probability. There are exactly 15 slots, so a player at p = 0.5 can be in or out depending on who else had a big season.
- SHAP values (in `explain.md` and the figures) show how far each feature moves a player's log-odds relative to an **average qualified
  player**, not relative to the 15th-best one. Read the **feature groups**, not single features. WS, OWS, VORP, BPM and PER are highly
  correlated, so credit among them is split arbitrarily, and some (for example `z_bpm`) even get a negative sign in the logistic model.
- What drives the model, by share of SHAP:
  - impact metrics (BPM/VORP/WS/PER): about 40-60%
  - scoring and box-score volume: about 20-25%; points per game is the single largest feature in the logistic model
  - availability and team success: about 10-15% each
  - voter history: small

## Limitations

- **What the model cannot see:** narrative, media market, injuries late in the season, positional scarcity on the pre-2024 ballots,
  and the precise 65-game rule. Its typical errors:
  - false positives: stars who missed many games in 2020-2023 (Harden 2021 with 44 games, Durant 2023 with 47). This suggests,
    but does not prove, that voters began penalising absences before the rule existed
  - misses: lower-VORP players on good teams (Siakam, Simmons, DeRozan)
- It is trained on end-of-season stats. Mid-season predictions (`--season 2027`) come from a different distribution.
- The holdout is small (6 seasons), so differences of a few players between models are not meaningful.
- The All-Star model ignores conferences and the fact that selection happens mid-season.
- Rates are not shrunk toward the mean. The minutes filter (1000+, pro-rated) is the small-sample control.

## What was and was not verified

- Verified: the real data was downloaded and validated, and the full pipeline ran on it. The numbers above come from that run.
  Charts were rendered to PNG and inspected.
- Offline tests use a **synthetic fixture** (`tests/fixture/`, fictional players and teams, generated by `tests/make_fixture.py`). They
  run only in a temporary directory, and no synthetic output is written to `reports/`.
- Not verified: the 2026 predictions against reality, because the labels aren't in this data copy.

## Files

| File | Purpose |
|---|---|
| `config.py` | paths, splits, thresholds, rule seasons, expected schema |
| `ingest.py` | cache-first download (5 retries, backoff), unzip, schema/season validation, `DOWNLOAD_INFO.txt` |
| `features.py` | player-season table, 2TM handling, team features, labels, voter history, season-relative features |
| `evaluate.py` | per-season top-k, 65-game mask, positional allocation, calibration helpers |
| `train.py` | baselines, CV tuning, final fits, holdout, secondary tasks, `models/*.joblib`, `reports/metrics.json` |
| `explain.py` | SHAP (Linear + Tree explainers), group importances, player waterfalls, `reports/explain.md` |
| `report.py` | `reports/report.md` and summary charts |
| `nba_fetch.py` | cache-first, resumable nba_api fetch (5 retries with backoff, 0.8 s pause, `--proxy`) |
| `nba_features.py` | nba_api cache -> features; B-Ref <-> NBA id crosswalk with games-played validation |
| `experiments.py` | v2 feature-group comparison (All-NBA, MVP), significance gate, fatigue analysis, `reports/v2_report.md` |
| `predict.py` | score any season with the v2 models; exact 65-game rule once game logs are cached |
| `run_pipeline.py` | everything in order |
| `tests_smoke.py` | 15 offline tests (name normalisation, token-order matching and the injury signal added after the live run): schema, 2TM, labels, leakage (incl. fatigue), jumps, teammates, fetch cache/resume/timeout, with/without, crosswalk, end-to-end incl. a fake nba_api cache |
| `tests/fake_nba.py` | fake nba_api endpoints built from the synthetic fixture |
