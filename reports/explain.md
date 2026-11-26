# SHAP explanations (All-NBA)

Explained on the 1604 qualified player-seasons of the holdout (2020-2025). Values are contributions to the model's log-odds (before the monotone Platt calibration).

Chosen model (by forward-chaining CV PR-AUC): **logreg**; best tree model: **random_forest**.

## Feature groups (share of total mean |SHAP|, summed within group)

Each column is normalised to sum to 1. The logistic model is in log-odds; TreeExplainer on a random forest is in probability units, so only the shares are comparable.

|                                    |   logreg |   random_forest |
|:-----------------------------------|---------:|----------------:|
| impact metrics (BPM/VORP/WS/PER)   |    0.402 |           0.607 |
| box-score volume (pts/reb/ast/...) |    0.257 |           0.182 |
| availability (games, minutes)      |    0.146 |           0.035 |
| team success                       |    0.144 |           0.082 |
| voter history (prior selections)   |    0.03  |           0.091 |
| age                                |    0.021 |           0.003 |


## Top 12 single features (mean |SHAP|)

|    | logreg            | random_forest     |
|---:|:------------------|:------------------|
|  0 | z_pts_per_game    | z_per             |
|  1 | z_mp_per_game     | z_ws              |
|  2 | rank_per          | z_vorp            |
|  3 | rank_pts_per_game | rank_ws           |
|  4 | z_dws             | z_pts_per_game    |
|  5 | z_ast_per_game    | rank_per          |
|  6 | team_win_rank     | rank_vorp         |
|  7 | rank_ws           | rank_pts_per_game |
|  8 | team_win_pct      | prev_all_nba      |
|  9 | z_per             | z_mp_per_game     |
| 10 | z_trb_per_game    | z_usg_percent     |
| 11 | z_usg_percent     | z_obpm            |


## Cross-checks

- Logistic model: Spearman between |coefficient| (standardised) and mean |SHAP| = **0.99**; coefficient sign matches the SHAP-vs-value direction for **100%** of features.
- Agreement between the logistic and random_forest importance rankings (Spearman) = **0.42**.
- Many inputs are strongly correlated (WS, OWS, VORP, BPM, PER all measure overall impact), so the credit is shared among them unpredictably. In the logistic model some of them even get a negative sign (for example `z_bpm`) once the others are included. Group totals are more stable than single-feature ranks.

## Three players

Contributions are measured against an *average qualified player* (the waterfall's E[f(X)]), not against the 15th-best player. An All-NBA candidate can be far above average on everything and still miss the cut. The p shown is the calibrated probability.

- **correct pick: Giannis Antetokounmpo 2025** (actual: All-NBA 1 team; 67 games, VORP 6.6, team win% 0.585) -> p = 1.00, rank 3. Largest contributions: pushed up by `z_pts_per_game` = 2.82 (+4.33), `z_per` = 3.60 (+1.29), `rank_per` = 3.00 (+1.26), `z_dws` = 1.94 (+1.21); pushed down by `rank_pts_per_game` = 2.00 (-1.16), `z_bpm` = 3.33 (-1.00), `z_dbpm` = 2.03 (-0.42), `prior_mvp_top5` = 6.00 (-0.36). Plot: `figures/shap_waterfall_correct_pick.png`
- **surprising miss: Ben Simmons 2020** (actual: All-NBA 3 team; 57 games, VORP 2.8, team win% 0.589) -> p = 0.11, rank 25. Largest contributions: pushed up by `z_ast_per_game` = 2.72 (+1.78), `z_mp_per_game` = 1.57 (+1.47), `z_pts_per_game` = 0.65 (+1.08), `z_dws` = 1.70 (+1.06); pushed down by `rank_pts_per_game` = 65.00 (-0.56), `z_bpm` = 1.32 (-0.39), `z_dbpm` = 1.82 (-0.37), `z_stl_per_game` = 3.43 (-0.23). Plot: `figures/shap_waterfall_surprising_miss.png`
- **false positive: James Harden 2021** (actual: not selected; 44 games, VORP 3.7, team win% 0.589) -> p = 0.94, rank 5. Largest contributions: pushed up by `z_pts_per_game` = 1.93 (+2.99), `z_ast_per_game` = 3.95 (+2.60), `z_mp_per_game` = 1.76 (+1.65), `rank_per` = 11.00 (+1.17); pushed down by `rank_pts_per_game` = 20.00 (-0.98), `z_bpm` = 2.51 (-0.75), `prior_mvp_top5` = 6.00 (-0.36), `g_share` = 0.61 (-0.18). Plot: `figures/shap_waterfall_false_positive.png`
