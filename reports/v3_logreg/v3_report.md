# v3: pre-registered feature sets (improvement plan 2024-25) - model family: logreg

Same harness as v2. Every set is tuned on the forward-chaining CV seasons 2005-2019 with the frozen v2 grids (H1c alone gets a wider logistic grid). `gain_vs_A` is the mean per-season PR-AUC change vs the 1980+ v1 feature set, with a paired season bootstrap CI Bonferroni-adjusted for 16 comparisons (8 sets x 2 passes: best family, and logistic regression only). **A set is adopted only if its whole CI is above 0.** 1997+ sets also show their gain vs B base (same window) as a diagnostic.

Holdout hits (2020-2025, 65-game mask) are shown for information only. H6-H8 were designed after looking at those seasons, so their holdout numbers are optimistic. The clean test is 2025-26.

| set                      |   n_features |   train_from | model   |   train_coverage |   cv_pr_auc |   gain_vs_A |   ci_low |   ci_high | adopt   | window_ref           |   gain_vs_window |   window_ci_low |   window_ci_high |   test_hits |   test_slots |
|:-------------------------|-------------:|-------------:|:--------|-----------------:|------------:|------------:|---------:|----------:|:--------|:---------------------|-----------------:|----------------:|-----------------:|------------:|-------------:|
| A base (v1 features)     |           34 |         1980 | logreg  |           1.0000 |      0.8996 |      0.0000 |   0.0000 |    0.0000 | False   | nan                  |         nan      |        nan      |         nan      |          73 |           90 |
| B base (v1 features)     |           34 |         1997 | logreg  |           1.0000 |      0.8879 |     -0.0120 |  -0.0276 |    0.0010 | False   | nan                  |         nan      |        nan      |         nan      |          74 |           90 |
| H1a team block           |           31 |         1980 | logreg  |           1.0000 |      0.8983 |     -0.0036 |  -0.0102 |    0.0018 | False   | A base (v1 features) |          -0.0036 |         -0.0102 |           0.0018 |          74 |           90 |
| H1b pts curvature        |           34 |         1980 | logreg  |           1.0000 |      0.8949 |     -0.0043 |  -0.0171 |    0.0060 | False   | A base (v1 features) |          -0.0043 |         -0.0171 |           0.0060 |          74 |           90 |
| H1c wider regularisation |           34 |         1980 | logreg  |           1.0000 |      0.9004 |     -0.0001 |  -0.0019 |    0.0013 | False   | A base (v1 features) |          -0.0001 |         -0.0019 |           0.0013 |          73 |           90 |
| H3 team role             |           36 |         1980 | logreg  |           1.0000 |      0.8964 |     -0.0056 |  -0.0107 |   -0.0004 | False   | A base (v1 features) |          -0.0056 |         -0.0107 |          -0.0004 |          73 |           90 |
| H4 seed                  |           36 |         1997 | logreg  |           1.0000 |      0.8871 |     -0.0135 |  -0.0290 |    0.0004 | False   | B base (v1 features) |          -0.0015 |         -0.0061 |           0.0021 |          74 |           90 |
| H6a archetype            |           46 |         1997 | logreg  |           1.0000 |      0.8919 |     -0.0065 |  -0.0228 |    0.0085 | False   | B base (v1 features) |           0.0055 |         -0.0067 |           0.0187 |          73 |           90 |
| H7 defense gate          |           38 |         1980 | logreg  |           1.0000 |      0.8974 |     -0.0041 |  -0.0090 |    0.0012 | False   | A base (v1 features) |          -0.0041 |         -0.0090 |           0.0012 |          73 |           90 |
| H8 improver              |           36 |         1980 | logreg  |           1.0000 |      0.8996 |      0.0004 |  -0.0118 |    0.0102 | False   | A base (v1 features) |           0.0004 |         -0.0118 |           0.0102 |          76 |           90 |


## Holdout hits per season

| set                      |   hits_2020 |   hits_2021 |   hits_2022 |   hits_2023 |   hits_2024 |   hits_2025 |
|:-------------------------|------------:|------------:|------------:|------------:|------------:|------------:|
| A base (v1 features)     |          10 |          13 |          13 |          11 |          13 |          13 |
| B base (v1 features)     |          12 |          13 |          12 |          11 |          13 |          13 |
| H1a team block           |          11 |          13 |          13 |          11 |          13 |          13 |
| H1b pts curvature        |          11 |          13 |          13 |          11 |          13 |          13 |
| H1c wider regularisation |          10 |          13 |          13 |          11 |          13 |          13 |
| H3 team role             |          11 |          13 |          12 |          11 |          13 |          13 |
| H4 seed                  |          12 |          13 |          12 |          11 |          13 |          13 |
| H6a archetype            |          12 |          12 |          12 |          11 |          13 |          13 |
| H7 defense gate          |          11 |          13 |          13 |          11 |          12 |          13 |
| H8 improver              |          13 |          13 |          13 |          11 |          13 |          13 |


**Served model:** A base (v1 features).

## Diagnostics (holdout; never used to choose)

| set                      |   rank_mitchell_2025 |   rank_jwilliams_2025 |   rank_morant_2022 |   pred_C_2024_25 |   actual_C_2024_25 |   holdout_misses |
|:-------------------------|---------------------:|----------------------:|-------------------:|-----------------:|-------------------:|-----------------:|
| A base (v1 features)     |                   16 |                    17 |                 17 |                8 |                  5 |               17 |
| B base (v1 features)     |                   17 |                    18 |                 16 |                8 |                  5 |               16 |
| H1a team block           |                   17 |                    15 |                 16 |                8 |                  5 |               16 |
| H1b pts curvature        |                   15 |                    16 |                 14 |                8 |                  5 |               16 |
| H1c wider regularisation |                   17 |                    16 |                 16 |                8 |                  5 |               17 |
| H3 team role             |                   18 |                    15 |                 17 |                8 |                  5 |               17 |
| H4 seed                  |                   17 |                    18 |                 16 |                8 |                  5 |               16 |
| H6a archetype            |                   15 |                    18 |                 14 |                8 |                  5 |               17 |
| H7 defense gate          |                   15 |                    18 |                 17 |                8 |                  5 |               17 |
| H8 improver              |                   14 |                    18 |                 15 |                8 |                  5 |               14 |


## H6c finisher-big rule (fixed, positionless seasons only)

A Rim-Running Big listed PF/C who is not his team's leading scorer has his rebounding, blocks, DWS and TS% z-scores capped at 0 (league average) in seasons from 2024. No parameter was tuned. 2024-25 informed the rule, so this table cannot validate it; only 2025-26 can.

| scorer                  |   hits_2020 |   hits_2021 |   hits_2022 |   hits_2023 |   hits_2024 |   hits_2025 |   total |
|:------------------------|------------:|------------:|------------:|------------:|------------:|------------:|--------:|
| served model            |          12 |          13 |          12 |          11 |          13 |          13 |      74 |
| served model + H6c rule |          12 |          13 |          12 |          11 |          14 |          14 |      76 |

