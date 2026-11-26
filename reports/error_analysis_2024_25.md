# Error Analysis: 2024–25 All-NBA Predictions

**Date:** 2025-11-15  
**Model:** Logistic Regression (v1, Platt-calibrated)  
**Training data:** 1980–2019 | **Holdout:** 2020–2025  
**Results:** 13/15 correct, measured against the Kaggle labels, which match the "Actual" column below. The repo has no source for a "12/15" figure.

> **Correction (2026-11-15):** The Root Cause Analysis, Key Insights ("What the Model Misses") and Recommendations sections below are contradicted by the data and the saved model:
> - `team_best_share` is **not** a v1 feature.
> - CLE's VORP leader is Mobley, not Allen.
> - Allen's BPM is 4.2, higher than Mitchell's 3.7.
> - `team_win_rank` (rank 1 = best) adds +0.83 logit to Mitchell; it does not hurt him.
> - `rank_per` adds +1.03.
> - The `prior_all_nba` coefficient is 0.097.
> - CLE is the **East** #1 seed, and OKC had the best record (0.829).
>
> The gap to the #15 player comes mostly from rebounds, blocks and DWS. See §1–2 of `improvement_plan_2024_25.md` for the corrections table and the verified analysis.

---

## Prediction Summary

### Predicted vs. Actual

| Rank | Player | Team | Predicted | Actual | Probability |
|------|--------|------|-----------|--------|-------------|
| 1 | Nikola Jokić | DEN | **1st** | **1st** | 1.000 |
| 2 | Shai Gilgeous-Alexander | OKC | **1st** | **1st** | 0.999 |
| 3 | Giannis Antetokounmpo | MIL | **1st** | **1st** | 0.995 |
| 4 | Jayson Tatum | BOS | **1st** | **1st** | 0.970 |
| 5 | LeBron James | LAL | **1st** | 2nd | 0.939 |
| 6 | Karl-Anthony Towns | NYK | 2nd | 3rd | 0.806 |
| 7 | Anthony Edwards | MIN | 2nd | 2nd | 0.783 |
| 8 | James Harden | LAC | 2nd | 3rd | 0.752 |
| 9 | Jalen Brunson | NYK | 2nd | 2nd | 0.616 |
| 10 | Domantas Sabonis | SAC | 2nd | — | 0.598 |
| 11 | Stephen Curry | GSW | 3rd | 2nd | 0.571 |
| 12 | Tyrese Haliburton | IND | 3rd | 3rd | 0.564 |
| 13 | Ivica Zubac | LAC | 3rd | — | 0.454 |
| 14 | Cade Cunningham | DET | 3rd | 3rd | 0.394 |
| 15 | Evan Mobley | CLE | 3rd | 2nd | 0.258 |

### The Two Players Missed

| Player | Actual Team | Model Rank | Probability | Key Issue |
|--------|-------------|------------|-------------|-----------|
| **Donovan Mitchell** | 1st Team | #17 | 0.206 | Not "best player" by VORP; PER rank 26th |
| **Jalen Williams** | 3rd Team | #16 | 0.215 | No prior All-NBA; PER rank 29th |

### False Positives (Predicted but Not Selected)

| Player | Model Rank | Probability | Why Model Liked Him |
|--------|------------|-------------|---------------------|
| Domantas Sabonis | #10 | 0.598 | Elite passing for a center (z_ast = +1.51), high VORP, prior All-NBA |
| Ivica Zubac | #13 | 0.454 | Super durable (80 games), efficient for a backup center |

---

## Root Cause Analysis

### 1. The `team_best_share` Feature Biases Against Guards

The model has a binary feature `team_best_share` that answers: **"Is this player the #1 player on their team by VORP?"**

For Donovan Mitchell on Cleveland:
- **Jarrett Allen's VORP: 3.6** (big man, rebounds, efficient)
- **Mitchell's VORP: 3.2** (guard, high usage, less rebounding)

Result: `team_best_share = 0.0` — the model thinks Mitchell is **not** the best player on his team.

But by every other metric, Mitchell is the clear #1 offensive weapon:

| Metric | Mitchell | Allen |
|--------|----------|-------|
| Points per game | 24.0 | 13.5 |
| Usage % | ~30% | ~15% |
| BPM | 3.7 | ~2.0 |
| True Shooting % | ~59% | ~62% |

**VORP heavily rewards rebounding and efficiency, which systematically favors big men.** The model's definition of "best player on team" undervalues elite guards.

### 2. `team_win_rank` Has a Negative Coefficient

The logistic regression coefficient for `team_win_rank` is **-0.5471**. This means:

- `team_win_rank = 1` (best team) → contribution = **-0.55**
- `team_win_rank = 20` → contribution = **-10.9**

After controlling for `team_win_pct` (+0.47), the rank feature acts as a **negative control**. Better teams get slightly less credit from this feature. So being on the #2 team in the NBA doesn't help Mitchell — it slightly hurts him compared to being on a mediocre team.

### 3. The Model Has No Seeding or Playoff Feature

There is **no feature for**:
- Conference seed (#1 seed in the West for CLE)
- Playoff position
- "Ace" status (best player by non-VORP metrics)
- Being the #2 player on a contender

Voters clearly value Mitchell's role as the #2 guy on the best team in the East. The model can't see that context.

### 4. PER Rank Is the Single Biggest Drag

Every 1st team player had a top-10 PER rank. Mitchell's was **26th** — by far his weakest stat:

| Player | PER Rank |
|--------|----------|
| Nikola Jokić | 3rd |
| Shai Gilgeous-Alexander | 2nd |
| Giannis Antetokounmpo | 5th |
| Jayson Tatum | 7th |
| LeBron James | 11th |
| **Donovan Mitchell** | **26th** |
| **Jalen Williams** | **29th** |

The model's #1 feature is `z_pts_per_game` (+1.50 coefficient), but points alone aren't enough — you need elite efficiency (high PER) to get the big boost. Mitchell's 24 PPG at moderate efficiency isn't compelling enough.

### 5. No Prior All-NBA History

The model rewards established reputation. Compare `prev_all_nba` (was on All-NBA last season?) and `prior_all_nba` (career All-NBA selections):

| Player | prev_all_nba | prior_all_nba |
|--------|-------------|---------------|
| Jokić | 1 | 6 |
| SGA | 1 | 2 |
| Giannis | 1 | 8 |
| Tatum | 1 | 4 |
| LeBron | 1 | 12 |
| **Mitchell** | **0** | **1** |
| **Jalen Williams** | **0** | **0** |

The model gives a +0.26 boost for `prev_all_nba` and +0.37 for `prior_all_nba`. Mitchell and Jalen Williams got zero from both.

---

## Feature Importance (Logistic Regression)

### Top 15 Features by Absolute Coefficient

| Rank | Feature | Coefficient | What It Measures |
|------|---------|-------------|------------------|
| 1 | z_pts_per_game | +1.50 | Scoring volume (season-relative) |
| 2 | z_mp_per_game | +0.94 | Minutes played |
| 3 | rank_per | -0.78 | PER rank (lower = better, so negative coef) |
| 4 | z_dws | +0.68 | Defensive win shares |
| 5 | rank_pts_per_game | +0.67 | Points rank |
| 6 | z_ast_per_game | +0.66 | Assists |
| 7 | team_win_rank | -0.55 | Team win rank (negative!) |
| 8 | rank_ws | -0.50 | Win shares rank |
| 9 | team_win_pct | +0.47 | Team winning percentage |
| 10 | z_per | +0.36 | PER |
| 11 | z_trb_per_game | +0.35 | Rebounds |
| 12 | z_usg_percent | +0.34 | Usage rate |
| 13 | z_bpm | -0.30 | BPM |
| 14 | prev_all_nba | +0.26 | Was on All-NBA last season |
| 15 | z_blk_per_game | +0.25 | Blocks |

### Feature Groups (by SHAP contribution)

| Group | Share of Total | What It Measures |
|-------|----------------|------------------|
| Impact metrics (BPM/VORP/WS/PER) | 40% | Overall player impact |
| Box-score volume (pts/reb/ast) | 26% | Scoring, rebounding, passing |
| Availability (games, minutes) | 15% | Durability, 65-game rule |
| Team success | 14% | Team win %, SRS, net rating |
| Voter history | 3% | Prior All-NBA, All-Star, MVP |
| Age | 2% | Player age |

---

## Key Insights

### What the Model Does Well
- **Dominant players are easy** — Jokić, SGA, Giannis, Tatum all had elite PER ranks and high VORP
- **Durability matters** — the 65-game rule is enforced exactly; players who missed games are excluded
- **Team quality helps** — `team_win_pct` gives a solid +0.47 boost
- **Reputation is rewarded** — prior All-NBA selections give a meaningful bump

### What the Model Misses
- **The "ace on a contender" effect** — being the best player (by most metrics) on a top team, even if VORP says otherwise
- **Conference seeding** — being the #2 seed in the NBA is valuable context voters use
- **Role on a great team** — being the #2 guy on a championship contender (Mitchell, Jalen Williams)
- **Narrative and eye-test** — media markets, playoff storylines, defensive reputation
- **VORP's big-man bias** — the model's definition of "best player" is skewed toward big men

### Why Sabonis and Zubac Were False Positives
- **Sabonis** had elite passing for a center (z_ast = +1.51), high VORP, and prior All-NBA history. But his team was bad (49% win rate) — voters didn't care.
- **Zubac** was super durable (80 games) and efficient for a backup center. But he's a role player, not a star — voters went with better players.

---

## Recommendations for Improvement

1. **Add a "best player by scoring" feature** — `team_pts_share` or `team_bpm_share` would capture Mitchell's status as Cleveland's #1 offensive weapon
2. **Add conference seeding** — `conf_seed` or `playoff_position` would give credit to players on top seeds
3. **Use multiple "best player" signals** — not just VORP, but also PPG share, BPM share, and PER share
4. **Add a "top-2 player" feature** — being one of the two best players on a top-3 team should help
5. **Consider position-adjusted metrics** — VORP undervalues guards; a position-neutral metric would help
