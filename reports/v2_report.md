# v2: new feature groups (MVP fatigue, teammates' health, jumps, team with/without, narratives)

## What the groups are

| group | features | source |
|---|---|---|
| fatigue | `prior_mvp_wins`, `mvp_last_season`, `mvp_streak`, `seasons_since_mvp`, `prev_mvp_share` | MVP labels, strictly prior seasons |
| jump | `d_z_vorp`, `d_z_bpm`, `d_z_ws_48`, `d_z_pts_per_game`, `prev_g_share` | season-relative z this season minus last season |
| teammates | `z_teammates_missed_value`, `teammate_star_games_missed`, `z_best_teammate_vorp_pace` | Kaggle per-team rows: games missed by rotation teammates (weighted by BPM x minutes), best teammate VORP pace |
| narrative | `all_star_this_season`, `team_win_improvement`, `team_best_share`, `team_vorp_share`, `stat_leader` | All-Star this season (chosen in February, before voting), team win% change, best player on team, share of team VORP, stat-title leader |
| onoff | `z_onoff_net`, `z_on_court_pm` | Basketball-Reference on/off net rating per 100 possessions (1997+) |
| with_without | `ww_win_diff_shrunk`, `win_pct_with` | nba_api game logs: team win% in games played vs missed (shrunk by games missed) |
| availability | `g_counted_share` | nba_api game logs: games with 20+ min (+ up to two 15-20 min), as share of team games |
| seed | `conf_seed`, `top2_seed` | nba_api standings: conference seed |
| nba_adv | `z_pie`, `z_net_rating` | nba_api advanced dashboard: PIE, net rating |
| clutch | `z_clutch_pts`, `z_clutch_pm` | nba_api clutch dashboard (last 5 min, within 5 pts) |
| est | `z_e_net_rating` | nba_api estimated net rating |
| highlights | `triple_doubles_rate`, `games_40pts_rate` | nba_api game logs: triple-doubles and 40-point games per team game |

There is no media/narrative data here, so the narrative features are measurable stand-ins for narrative, not narrative itself.

## nba_api data status

- Endpoints and seasons cached: {'clutch': '1997-2027 (31)', 'dash_adv': '1997-2027 (31)', 'est_metrics': '1997-2027 (31)', 'gamelog_P': '1997-2027 (31)', 'gamelog_T': '1997-2027 (31)', 'standings': '1997-2027 (31)'}
- Crosswalk B-Ref -> NBA ids: {'bref_rows': 15010, 'matched': 15010, 'unmatched': 0, 'match_rate': 1.0, 'methods': {'exact': 14942, 'fuzzy': 60, 'exact_tiebreak': 8}, 'games_mismatch_gt2': 337} (unmatched rows: `crosswalk_unmatched.csv`).

## All-NBA: does each group help?

Every row is tuned and scored on the same forward-chaining CV seasons (15 seasons, 2005-2019). `cv_ap_gain_vs_v1` is the mean per-season PR-AUC change vs the v1 feature set, with a paired bootstrap CI over seasons, Bonferroni-adjusted for 16 comparisons. A set is adopted only if the whole CI is above 0. Holdout hits (2020+, 65-game mask) are shown for information and were never used to choose.

`gain_vs_window_ref` tests each group fairly inside its own training window (1997+ sets against the 1997+ set without that group). `adopt` is the production bar: the set must also beat the 1980+ v1 model.

| set                                |   n_features |   train_from | model   |   cv_pr_auc |   cv_precision_at_k |   test_pr_auc_g |   test_hits_g |   test_slots |   test_pr_auc_exact |   test_hits_exact |   cv_ap_gain_vs_v1 |   gain_ci_low |   gain_ci_high | adopt   | window_ref           |   gain_vs_window_ref |   window_ci_low |   window_ci_high | group_helps_in_window   |
|:-----------------------------------|-------------:|-------------:|:--------|------------:|--------------------:|----------------:|--------------:|-------------:|--------------------:|------------------:|-------------------:|--------------:|---------------:|:--------|:---------------------|---------------------:|----------------:|-----------------:|:------------------------|
| A base (v1 features)               |           34 |         1980 | logreg  |       0.900 |               0.809 |           0.917 |            73 |           90 |               0.917 |                73 |              0.000 |         0.000 |          0.000 | False   | nan                  |              nan     |         nan     |          nan     | False                   |
| A +fatigue                         |           39 |         1980 | hist_gb |       0.898 |               0.818 |           0.901 |            72 |           90 |               0.901 |                72 |             -0.008 |        -0.027 |          0.012 | False   | A base (v1 features) |               -0.008 |          -0.027 |            0.012 | False                   |
| A +jump                            |           39 |         1980 | logreg  |       0.898 |               0.800 |           0.920 |            74 |           90 |               0.920 |                74 |             -0.004 |        -0.015 |          0.008 | False   | A base (v1 features) |               -0.004 |          -0.015 |            0.008 | False                   |
| A +teammates                       |           37 |         1980 | hist_gb |       0.900 |               0.809 |           0.889 |            72 |           90 |               0.889 |                72 |             -0.003 |        -0.022 |          0.014 | False   | A base (v1 features) |               -0.003 |          -0.022 |            0.014 | False                   |
| A +narrative                       |           39 |         1980 | hist_gb |       0.905 |               0.822 |           0.902 |            71 |           90 |               0.902 |                71 |             -0.002 |        -0.027 |          0.024 | False   | A base (v1 features) |               -0.002 |          -0.027 |            0.024 | False                   |
| A +all Kaggle groups               |           52 |         1980 | hist_gb |       0.901 |               0.818 |           0.897 |            73 |           90 |               0.897 |                73 |             -0.007 |        -0.034 |          0.025 | False   | A base (v1 features) |               -0.007 |          -0.034 |            0.025 | False                   |
| B base (v1 features)               |           34 |         1997 | hist_gb |       0.895 |               0.804 |           0.920 |            72 |           90 |               0.920 |                72 |             -0.007 |        -0.028 |          0.013 | False   | nan                  |              nan     |         nan     |          nan     | False                   |
| B +all Kaggle groups               |           52 |         1997 | logreg  |       0.896 |               0.809 |           0.905 |            76 |           90 |               0.905 |                76 |             -0.005 |        -0.035 |          0.021 | False   | B base (v1 features) |                0.003 |          -0.031 |            0.036 | False                   |
| B +all Kaggle +onoff               |           54 |         1997 | logreg  |       0.897 |               0.809 |           0.903 |            76 |           90 |               0.903 |                76 |             -0.002 |        -0.030 |          0.020 | False   | B base (v1 features) |                0.005 |          -0.027 |            0.036 | False                   |
| B +all Kaggle +onoff +with_without |           56 |         1997 | logreg  |       0.900 |               0.818 |           0.900 |            74 |           90 |               0.900 |                74 |             -0.000 |        -0.028 |          0.023 | False   | B +all Kaggle +onoff |                0.002 |          -0.004 |            0.008 | False                   |
| B +all Kaggle +onoff +availability |           55 |         1997 | hist_gb |       0.898 |               0.800 |           0.914 |            75 |           90 |               0.914 |                75 |             -0.009 |        -0.031 |          0.014 | False   | B +all Kaggle +onoff |               -0.007 |          -0.032 |            0.021 | False                   |
| B +all Kaggle +onoff +seed         |           56 |         1997 | logreg  |       0.896 |               0.809 |           0.903 |            75 |           90 |               0.903 |                75 |             -0.006 |        -0.031 |          0.016 | False   | B +all Kaggle +onoff |               -0.004 |          -0.009 |            0.000 | False                   |
| B +all Kaggle +onoff +nba_adv      |           56 |         1997 | logreg  |       0.897 |               0.809 |           0.903 |            76 |           90 |               0.903 |                76 |             -0.002 |        -0.031 |          0.022 | False   | B +all Kaggle +onoff |                0.000 |          -0.004 |            0.004 | False                   |
| B +all Kaggle +onoff +clutch       |           56 |         1997 | hist_gb |       0.901 |               0.800 |           0.907 |            74 |           90 |               0.907 |                74 |             -0.006 |        -0.026 |          0.015 | False   | B +all Kaggle +onoff |               -0.004 |          -0.028 |            0.025 | False                   |
| B +all Kaggle +onoff +est          |           55 |         1997 | hist_gb |       0.898 |               0.804 |           0.901 |            72 |           90 |               0.901 |                72 |             -0.007 |        -0.028 |          0.015 | False   | B +all Kaggle +onoff |               -0.005 |          -0.030 |            0.021 | False                   |
| B +all Kaggle +onoff +highlights   |           56 |         1997 | hist_gb |       0.900 |               0.813 |           0.914 |            76 |           90 |               0.914 |                76 |             -0.003 |        -0.024 |          0.022 | False   | B +all Kaggle +onoff |               -0.000 |          -0.025 |            0.026 | False                   |
| B +everything                      |           66 |         1997 | hist_gb |       0.898 |               0.809 |           0.913 |            77 |           90 |               0.913 |                77 |             -0.009 |        -0.028 |          0.010 | False   | B +all Kaggle +onoff |               -0.007 |          -0.033 |            0.025 | False                   |


**Verdict:** chosen v2 All-NBA set = **A base (v1 features)**. No new group improved All-NBA prediction beyond noise. The v1 features (box-score impact, scoring, availability, team success) already capture what these groups add.

## MVP: does each group help?

HistGradientBoosting regression on vote share. `cv_winner_rank_gain` = how many places higher the real winner is ranked than with the base features (rank capped at 10, mean over 15 CV seasons, Bonferroni-adjusted bootstrap CI). The cap matters: 1997+ sets train their first CV block on only 8 seasons, and that model ranks Steve Nash's atypical 2005-06 MVPs in the 200s; `worst_winner_rank` shows these outliers.

| set                  |   n_features |   cv_winner_hits |   cv_seasons |   cv_mean_winner_rank_capped |   cv_median_winner_rank |   cv_worst_winner_rank |   cv_mean_top5_overlap |   test_winner_hits |   test_seasons |   test_mean_winner_rank_capped |   test_median_winner_rank |   test_worst_winner_rank |   test_mean_top5_overlap |   cv_winner_rank_gain |   gain_ci_low |   gain_ci_high | adopt   | window_ref           |   gain_vs_window_ref |   window_ci_low | group_helps_in_window   |
|:---------------------|-------------:|-----------------:|-------------:|-----------------------------:|------------------------:|-----------------------:|-----------------------:|-------------------:|---------------:|-------------------------------:|--------------------------:|-------------------------:|-------------------------:|----------------------:|--------------:|---------------:|:--------|:---------------------|---------------------:|----------------:|:------------------------|
| A base               |           34 |                8 |           15 |                        3.133 |                   1.000 |                     18 |                  3.267 |                  4 |              6 |                          1.333 |                     1.000 |                        2 |                    3.833 |                 0.000 |         0.000 |          0.000 | False   | nan                  |              nan     |         nan     | False                   |
| A +fatigue           |           39 |                9 |           15 |                        3.000 |                   1.000 |                     20 |                  3.667 |                  4 |              6 |                          1.333 |                     1.000 |                        2 |                    3.833 |                 0.133 |        -0.400 |          0.867 | False   | A base               |                0.133 |          -0.400 | False                   |
| A +narrative         |           39 |                9 |           15 |                        3.200 |                   1.000 |                     27 |                  3.600 |                  4 |              6 |                          1.333 |                     1.000 |                        2 |                    3.833 |                -0.067 |        -0.400 |          0.200 | False   | A base               |               -0.067 |          -0.400 | False                   |
| A +all Kaggle groups |           52 |                9 |           15 |                        2.933 |                   1.000 |                     18 |                  3.467 |                  5 |              6 |                          1.500 |                     1.000 |                        4 |                    3.833 |                 0.200 |        -0.200 |          0.867 | False   | A base               |                0.200 |          -0.200 | False                   |
| B base               |           34 |                7 |           15 |                        2.800 |                   2.000 |                    248 |                  3.667 |                  4 |              6 |                          1.333 |                     1.000 |                        2 |                    3.500 |                 0.333 |        -0.200 |          1.067 | False   | nan                  |              nan     |         nan     | False                   |
| B +all Kaggle +onoff |           54 |                8 |           15 |                        2.733 |                   1.000 |                     17 |                  3.467 |                  5 |              6 |                          1.500 |                     1.000 |                        4 |                    3.500 |                 0.400 |        -0.133 |          1.267 | False   | B base               |                0.067 |          -0.333 | False                   |
| B +everything        |           66 |                9 |           15 |                        2.600 |                   1.000 |                    258 |                  3.867 |                  4 |              6 |                          1.500 |                     1.000 |                        3 |                    3.667 |                 0.533 |        -0.133 |          1.400 | False   | B +all Kaggle +onoff |                0.133 |           0.000 | False                   |


**Verdict:** chosen v2 MVP set = **A base** (nothing passed the adjusted CI; differences of one winner in 15 seasons are noise).

## MVP voter fatigue: is there any?

Reigning MVPs compared with their predicted share. A **negative** residual (real share below predicted) would mean fatigue. `won` vs `model_top_pick` compares how often reigning MVPs actually won with how often the model ranked them first.

| model      | data         | group                      |   n |   mean_real_share |   mean_pred_share |   mean_residual |   won |   model_top_pick |
|:-----------|:-------------|:---------------------------|----:|------------------:|------------------:|----------------:|------:|-----------------:|
| A base     | CV 2005-2019 | reigning MVP               |  14 |             0.510 |             0.420 |           0.090 |     4 |                4 |
| A base     | CV 2005-2019 | other top-10 by real share | 145 |             0.219 |             0.219 |          -0.000 |    11 |               11 |
| A base     | holdout      | reigning MVP               |   5 |             0.727 |             0.563 |           0.164 |     2 |                2 |
| A base     | holdout      | other top-10 by real share |  58 |             0.206 |             0.186 |           0.020 |     4 |                4 |
| A +fatigue | CV 2005-2019 | reigning MVP               |  14 |             0.510 |             0.436 |           0.074 |     4 |                4 |
| A +fatigue | CV 2005-2019 | other top-10 by real share | 145 |             0.219 |             0.222 |          -0.003 |    11 |               11 |
| A +fatigue | holdout      | reigning MVP               |   5 |             0.727 |             0.484 |           0.242 |     2 |                2 |
| A +fatigue | holdout      | other top-10 by real share |  58 |             0.206 |             0.183 |           0.023 |     4 |                4 |


**No evidence of voter fatigue.** Reigning MVPs received *more* vote share than their stats predict (mean residual +0.090), and they won 4 times vs 4 model top picks. Small samples: 14 reigning-MVP seasons in CV.

Season by season (base model):

|   season | data    | reigning_mvp          | winner                  | model_top_pick          | reigning_won   | model_picked_reigning   |
|---------:|:--------|:----------------------|:------------------------|:------------------------|:---------------|:------------------------|
|     2005 | CV      | Kevin Garnett         | Steve Nash              | Kevin Garnett           | False          | True                    |
|     2006 | CV      | Steve Nash            | Steve Nash              | LeBron James            | True           | False                   |
|     2007 | CV      | Steve Nash            | Dirk Nowitzki           | Dirk Nowitzki           | False          | False                   |
|     2008 | CV      | Dirk Nowitzki         | Kobe Bryant             | Chris Paul              | False          | False                   |
|     2009 | CV      | Kobe Bryant           | LeBron James            | LeBron James            | False          | False                   |
|     2010 | CV      | LeBron James          | LeBron James            | LeBron James            | True           | True                    |
|     2011 | CV      | LeBron James          | Derrick Rose            | LeBron James            | False          | True                    |
|     2012 | CV      | Derrick Rose          | LeBron James            | LeBron James            | False          | False                   |
|     2013 | CV      | LeBron James          | LeBron James            | Kevin Durant            | True           | False                   |
|     2014 | CV      | LeBron James          | Kevin Durant            | Kevin Durant            | False          | False                   |
|     2015 | CV      | (not qualified)       | Stephen Curry           | James Harden            | False          | False                   |
|     2016 | CV      | Stephen Curry         | Stephen Curry           | Stephen Curry           | True           | True                    |
|     2017 | CV      | Stephen Curry         | Russell Westbrook       | James Harden            | False          | False                   |
|     2018 | CV      | Russell Westbrook     | James Harden            | James Harden            | False          | False                   |
|     2019 | CV      | James Harden          | Giannis Antetokounmpo   | Giannis Antetokounmpo   | False          | False                   |
|     2020 | holdout | Giannis Antetokounmpo | Giannis Antetokounmpo   | Giannis Antetokounmpo   | True           | True                    |
|     2021 | holdout | Giannis Antetokounmpo | Nikola Jokić            | Nikola Jokić            | False          | False                   |
|     2022 | holdout | Nikola Jokić          | Nikola Jokić            | Joel Embiid             | True           | False                   |
|     2023 | holdout | Nikola Jokić          | Joel Embiid             | Nikola Jokić            | False          | True                    |
|     2024 | holdout | (not qualified)       | Nikola Jokić            | Nikola Jokić            | False          | False                   |
|     2025 | holdout | Nikola Jokić          | Shai Gilgeous-Alexander | Shai Gilgeous-Alexander | False          | False                   |


Reigning MVPs the model picked who then lost (the fatigue-looking cases): Kevin Garnett 2005 (won by Steve Nash), LeBron James 2011 (won by Derrick Rose), Nikola Jokić 2023 (won by Joel Embiid). Reigning MVPs the model did not pick who won anyway: Steve Nash 2006, LeBron James 2013, Nikola Jokić 2022. Both kinds occur (3 vs 3), so there is no consistent fatigue pattern.

## What the chosen v2 All-NBA model relies on (share of mean |SHAP| by group, holdout)

| group                              |   share |
|:-----------------------------------|--------:|
| impact metrics (BPM/VORP/WS/PER)   |   0.402 |
| box-score volume (pts/reb/ast/...) |   0.257 |
| availability (games, minutes)      |   0.146 |
| team success                       |   0.144 |
| voter history (prior selections)   |   0.030 |
| age                                |   0.021 |


