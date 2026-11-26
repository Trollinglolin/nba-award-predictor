# v3 opt-in candidate

```
Opt-in v3 candidate, fixed 2026-11-15 from three hypotheses, NOT validated by CV.
- Features: v1 features + H7 defense gate (def_pos, def_neg, offensive_ace, def_neg_x_ace)
            + H8 improver (team_scoring_leader, lead_improver).
- Model: logistic regression, C chosen on the frozen v2 grid by forward-chaining CV 2005-2019, trained 1980-2019,
  Platt-calibrated on the out-of-fold predictions.
- Scoring rule H6c, seasons >= 2024 only: a Rim-Running Big listed PF/C who is not his team's leading scorer has
  z_trb_per_game, z_blk_per_game, z_dws and z_ts_percent capped at 0 before scoring.
- Test: 2025-26 labels, scored once against v1. +/-1 hit out of 15 is noise.
```

CV (2005-2019) per-season PR-AUC gain vs v1 features, same family and grid: **-0.0010**, 95% CI [-0.0101, +0.0073] (single uncorrected comparison; exploratory). Tuned C: {"C": 0.1}.

Holdout hits (2020-2025, 65-game mask). H7, H8 and H6c were designed after looking at these seasons, so the candidate's numbers here are optimistic and cannot validate it:

|                      |   2020 |   2021 |   2022 |   2023 |   2024 |   2025 |   total |
|:---------------------|-------:|-------:|-------:|-------:|-------:|-------:|--------:|
| v1 (served)          |     12 |     13 |     12 |     11 |     13 |     13 |      74 |
| candidate, no rule   |     13 |     13 |     13 |     11 |     13 |     13 |      76 |
| candidate + H6c rule |     13 |     13 |     13 |     11 |     13 |     13 |      76 |

