# v3: pre-registered feature sets (improvement plan 2024-25) - model family: best of logreg/hist_gb

Same harness as v2. Every set is tuned on the forward-chaining CV seasons 2005-2019 with the frozen v2 grids (H1c alone gets a wider logistic grid). `gain_vs_A` is the mean per-season PR-AUC change vs the 1980+ v1 feature set, with a paired season bootstrap CI Bonferroni-adjusted for 16 comparisons (8 sets x 2 passes: best family, and logistic regression only). **A set is adopted only if its whole CI is above 0.** 1997+ sets also show their gain vs B base (same window) as a diagnostic.

Holdout hits (2020-2025, 65-game mask) are shown for information only. H6-H8 were designed after looking at those seasons, so their holdout numbers are optimistic. The clean test is 2025-26.

| set                      |   n_features |   train_from | model   |   train_coverage |   cv_pr_auc |   gain_vs_A |   ci_low |   ci_high | adopt   | window_ref           |   gain_vs_window |   window_ci_low |   window_ci_high |   test_hits |   test_slots |
|:-------------------------|-------------:|-------------:|:--------|-----------------:|------------:|------------:|---------:|----------:|:--------|:---------------------|-----------------:|----------------:|-----------------:|------------:|-------------:|
| A base (v1 features)     |           34 |         1980 | logreg  |           1.0000 |      0.8996 |      0.0000 |   0.0000 |    0.0000 | False   | nan                  |         nan      |        nan      |         nan      |          73 |           90 |
| B base (v1 features)     |           34 |         1997 | hist_gb |           1.0000 |      0.8945 |     -0.0072 |  -0.0279 |    0.0134 | False   | nan                  |         nan      |        nan      |         nan      |          72 |           90 |
| H1a team block           |           31 |         1980 | hist_gb |           1.0000 |      0.9001 |     -0.0053 |  -0.0275 |    0.0158 | False   | A base (v1 features) |          -0.0053 |         -0.0275 |           0.0158 |          74 |           90 |
| H1b pts curvature        |           34 |         1980 | hist_gb |           1.0000 |      0.8979 |     -0.0089 |  -0.0248 |    0.0065 | False   | A base (v1 features) |          -0.0089 |         -0.0248 |           0.0065 |          73 |           90 |
| H1c wider regularisation |           34 |         1980 | logreg  |           1.0000 |      0.9004 |     -0.0001 |  -0.0019 |    0.0013 | False   | A base (v1 features) |          -0.0001 |         -0.0019 |           0.0013 |          73 |           90 |
| H3 team role             |           36 |         1980 | hist_gb |           1.0000 |      0.9007 |     -0.0053 |  -0.0283 |    0.0170 | False   | A base (v1 features) |          -0.0053 |         -0.0283 |           0.0170 |          75 |           90 |
| H4 seed                  |           36 |         1997 | hist_gb |           1.0000 |      0.8923 |     -0.0154 |  -0.0367 |    0.0060 | False   | B base (v1 features) |          -0.0082 |         -0.0235 |           0.0038 |          74 |           90 |
| H6a archetype            |           46 |         1997 | hist_gb |           1.0000 |      0.8926 |     -0.0121 |  -0.0349 |    0.0087 | False   | B base (v1 features) |          -0.0049 |         -0.0163 |           0.0065 |          74 |           90 |
| H7 defense gate          |           38 |         1980 | hist_gb |           1.0000 |      0.8997 |     -0.0055 |  -0.0246 |    0.0201 | False   | A base (v1 features) |          -0.0055 |         -0.0246 |           0.0201 |          74 |           90 |
| H8 improver              |           36 |         1980 | logreg  |           1.0000 |      0.8996 |      0.0004 |  -0.0118 |    0.0102 | False   | A base (v1 features) |           0.0004 |         -0.0118 |           0.0102 |          76 |           90 |


## Holdout hits per season

| set                      |   hits_2020 |   hits_2021 |   hits_2022 |   hits_2023 |   hits_2024 |   hits_2025 |
|:-------------------------|------------:|------------:|------------:|------------:|------------:|------------:|
| A base (v1 features)     |          10 |          13 |          13 |          11 |          13 |          13 |
| B base (v1 features)     |          11 |          11 |          12 |          11 |          14 |          13 |
| H1a team block           |          12 |          12 |          13 |          12 |          13 |          12 |
| H1b pts curvature        |          12 |          11 |          13 |          11 |          14 |          12 |
| H1c wider regularisation |          10 |          13 |          13 |          11 |          13 |          13 |
| H3 team role             |          12 |          11 |          14 |          12 |          14 |          12 |
| H4 seed                  |          12 |          11 |          13 |          12 |          13 |          13 |
| H6a archetype            |          11 |          12 |          12 |          13 |          14 |          12 |
| H7 defense gate          |          12 |          12 |          12 |          11 |          14 |          13 |
| H8 improver              |          13 |          13 |          13 |          11 |          13 |          13 |


**Served model:** A base (v1 features).

## Diagnostics (holdout; never used to choose)

| set                      |   rank_mitchell_2025 |   rank_jwilliams_2025 |   rank_morant_2022 |   pred_C_2024_25 |   actual_C_2024_25 |   holdout_misses |
|:-------------------------|---------------------:|----------------------:|-------------------:|-----------------:|-------------------:|-----------------:|
| A base (v1 features)     |                   16 |                    17 |                 17 |                8 |                  5 |               17 |
| B base (v1 features)     |                   11 |                    18 |                 16 |                7 |                  5 |               18 |
| H1a team block           |                   15 |                    16 |                 16 |                7 |                  5 |               16 |
| H1b pts curvature        |                   10 |                    14 |                 16 |                7 |                  5 |               17 |
| H1c wider regularisation |                   17 |                    16 |                 16 |                8 |                  5 |               17 |
| H3 team role             |                   15 |                    17 |                 15 |                7 |                  5 |               15 |
| H4 seed                  |                   13 |                    15 |                 13 |                7 |                  5 |               16 |
| H6a archetype            |                   14 |                    16 |                 17 |                8 |                  5 |               16 |
| H7 defense gate          |                   14 |                    12 |                 18 |                6 |                  5 |               16 |
| H8 improver              |                   14 |                    18 |                 15 |                8 |                  5 |               14 |


## H6c finisher-big rule (fixed, positionless seasons only)

A Rim-Running Big listed PF/C who is not his team's leading scorer has his rebounding, blocks, DWS and TS% z-scores capped at 0 (league average) in seasons from 2024. No parameter was tuned. 2024-25 informed the rule, so this table cannot validate it; only 2025-26 can.

| scorer                  |   hits_2020 |   hits_2021 |   hits_2022 |   hits_2023 |   hits_2024 |   hits_2025 |   total |
|:------------------------|------------:|------------:|------------:|------------:|------------:|------------:|--------:|
| served model            |          12 |          13 |          12 |          11 |          13 |          13 |      74 |
| served model + H6c rule |          12 |          13 |          12 |          11 |          14 |          14 |      76 |

