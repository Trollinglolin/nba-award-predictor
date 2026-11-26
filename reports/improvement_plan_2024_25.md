# Improvement Plan: 2024–25 All-NBA Misses (revised)

**Date:** 2026-11-15. This revises the 2025-11-15 draft, which is kept as `improvement_plan_2024_25.orig.md`.
**Based on:** `reports/error_analysis_2024_25.md`, checked against `data/processed/player_seasons.csv`, the saved model `models/all_nba_logreg.joblib`, `reports/test_predictions.csv`, `reports/v2_all_nba_sets.csv` and `experiments.run_all_nba`.
**Current model:** v1 logistic regression (C=0.1, StandardScaler, Platt-calibrated), 34 features, trained 1980–2019.
**Holdout:** 2020–2025.
- 73/90 hits in the v2 harness and 74/90 in `test_per_season.csv` (eligibility mask).
- 2025 scored 13/15, above the holdout average of about 12.2.

---

## 0. What changed and why

The first draft blamed `team_best_share`, a VORP-based "best player on team" flag, and built five fixes on that diagnosis. The data and the trained model do not support it:

- **`team_best_share` is not in the production model.** It belongs to the v2 `narrative` group, which was tested and not adopted (CV gain −0.002, CI [−0.027, +0.024]).
- **Both sign readings were backwards.** `team_win_rank` and `rank_per` raise Mitchell's score; they do not lower it.
- **The misses were near-misses.** Jalen Williams ranked #16 (p=0.215) and Mitchell #17 (p=0.206), just below #15 (Mobley, p=0.258).

Two findings do hold up:

1. **Collinear, sign-flipped features.**
2. **Too many centers in the positionless era.** This affects 2024–25 only: 3 false-positive centers (Gobert, Sabonis, Zubac). It cannot be validated with 1980–2019 data, because those seasons all used positional ballots.

The draft's experiment design also chose features using the holdout players, which this revision removes.

## 1. Corrections to the original analysis and plan

| # | Original claim | What the data shows | Evidence |
|---|---|---|---|
| 1 | `team_best_share = 0` is "one of the biggest probability drags" on Mitchell. | The feature is **not one of the 34 v1 features**, so it has no effect on the prediction. | `all_nba_logreg.joblib["features"]`, `config.FEATURES` |
| 2 | Allen is CLE's VORP leader. | **Mobley** has `team_best_share = 1`. VORP is pace-adjusted per game, and Mobley played 71 games to Allen's 82 at the same VORP of 3.6. | `player_seasons.csv`, 2025 |
| 3 | Allen's BPM is about 2.0 and Mitchell's 3.7. | **Allen 4.2, Mobley 4.6, Mitchell 3.7.** Mitchell leads CLE only in points and usage. | same |
| 4 | Mitchell would get `team_pts_share = 1` **and** `team_bpm_share = 1`. | He would get points only. `team_bpm_share = 0` and `ace_majority` (at least 2 of 3) **= 0**. | same |
| 5 | `team_win_rank` (coefficient −0.55) penalises good teams. | Rank 1 is the best team (`rank(ascending=False)`), and the coefficient applies to a standardised feature. A **negative coefficient therefore favours good teams**: it adds **+0.83 logit to Mitchell** and +0.90 to J. Williams. The "−10.9 at rank 20" arithmetic ignores the standardisation. | `features.team_table`; contribution = scaled x × coef |
| 6 | PER rank 26th is "the single biggest drag". | `rank_per` adds **+1.03** to Mitchell. Rank 26 is far above the average qualified player. | same |
| 7 | `prior_all_nba` adds +0.37. | Its coefficient is **+0.097**. | model coefficients |
| 8 | CLE was the "#1 seed in the West" or "#2 seed". | CLE was the **East #1 seed** with the second-best record (0.780). OKC was **West #1 with the best record (0.829)**, not "57%, #2 seed". | `team_win_pct`, `conf_seed` |
| 9 | A `win_pct ≥ 0.610` threshold marks the top 3 teams. | 0.610 was the **6th**-best record in 2025 (LAL). The top-3 cutoff was BOS at 0.744. | same |
| 10 | The `+seed` set was "second-best" with 75/90. | **Five** sets scored higher: four at 76 and one at 77, including its own reference. Its gain within its training window was **−0.0035, CI [−0.0087, +0.0002]**, which is borderline harmful. Picking by holdout hits is what the harness was built to avoid. | `v2_all_nba_sets.csv` |
| 11 | 12/15 against "real NBA results" vs 13/15 against Kaggle labels. | No "12/15" appears anywhere in the repo's `.py` or `.md` files. The Kaggle labels agree with the report's own "Actual" column for every player, which gives **13/15**. | `all_nba_team`; repo grep |
| 12 | Zubac is a "backup center". | Zubac started all 80 games he played (gs = 80). He was a false positive because of DWS and rebounding (§2). | `gs` |
| 13 | The errors are "systematic" because of the VORP-based definition of ace. | No v1 feature uses that definition. See §2 for the patterns that do recur. | — |

## 2. What actually explains the misses

### 2a. Gap decomposition

The table compares logit contributions: Mitchell or J. Williams minus the player ranked above them.

| Comparison | Total gap | Largest terms against | Largest terms for |
|---|---|---|---|
| Mitchell − Mobley (#15) | **−0.30** | blk −0.78, dws −0.70, trb −0.70, rank_pts −0.31 | pts +1.35, ast +0.61, usg +0.47 |
| Mitchell − Zubac (#13) | −1.22 | dws −1.33, trb −1.18, blk −0.50 | pts +1.77, ast +0.77, team_win_pct +0.53 |
| J. Williams − Zubac | −1.16 | trb −1.06, dws −0.42, team_srs −0.35 | pts +1.18, ast +0.81, team_win_pct +0.68 |
| J. Williams − Sabonis (#10) | −1.77 | trb −1.25, prev_all_nba −1.16, team_srs −0.54 | team_win_pct +1.06, team_win_rank +1.02, dws +0.98 |

Mitchell missed Mobley by 0.30 logit. Most of each gap comes from **rebounds, blocks and defensive win shares**, not from team success or any "best player on team" signal. Team success already pulls toward both Mitchell and Williams.

### 2b. Center bias depends on the era, and CV cannot see it

The model ranks every season the same way: all top-15 picks use the eligibility mask, and for seasons before 2024 the "positional" column also applies the slot quotas from `evaluate.top_k`. Rows marked in-sample use the saved model on its own training years.

| Seasons | Selection | Predicted C | Actual C | C missed | C false positives |
|---|---|---|---|---|---|
| 1990–2004 (in-sample) | positional | 45 | 48 | 8 | 5 |
| 2005–2019 (in-sample, CV window) | positional | 45 | **53** | **15** | 7 |
| 2020–2023 (holdout) | positional | 12 | 12 | 1 | 1 |
| 2024–2025 (holdout, positionless) | top-15 | **8** | **5** | 0 | **3** |

- Under positional ballots the model **under**-selects centers in 2005–2019 and gets them right in 2020–23.
- The surplus shows up only once the quota is gone, in 2024–25 (Gobert, Sabonis, Zubac). That is 3 players over 2 seasons.
- All training seasons had a quota of about 3 centers. The model therefore learned "big-man stat line → selected" from a process that partly guaranteed that outcome. Without the quota, voters pick fewer centers than those stats predict.
- **A position-adjusted feature tuned on 2005–2019 CV would be pushed the wrong way.** CV in that window rewards more centers, not fewer.
- Caveat: positions here come from B-Ref `pos`, which differ from ballot positions; for example, Mobley is listed as PF.

### 2c. Collinearity produces sign-flipped coefficients

The four team features are close to one variable: their pairwise correlations in training are 0.93–0.996 in absolute value. Several coefficients take the "wrong" sign:

| Feature | Coefficient | Partner it offsets |
|---|---|---|
| `rank_pts_per_game` | +0.67 (a worse scorer gets a higher score) | `z_pts_per_game` +1.50 |
| `team_srs` | −0.20 | `team_win_pct` +0.47, `team_win_rank` −0.55 |
| `z_bpm`, `z_dbpm` | −0.30, −0.23 | `z_vorp`, `z_dws`, rank_* |

`rank_pts_per_game` is the largest single negative term for every miss and false positive inspected (−0.92 for Mitchell). It may be encoding curvature: the z-score rewards scoring linearly, and the rank pulls back the very top.

This is a **hypothesis about interpretability**. It has not been shown to cost hits, so CV has to decide.

## 3. Revised improvements, ranked by evidence

Each improvement is a separate, pre-registered feature set (§4).

### H1. Collinearity cleanup

This replaces the old "drop `team_win_rank`". That proposal rested on a misreading of the sign, and the feature currently adds to Mitchell's score. Instead, test the team block and the duplicated scoring signal as blocks:

- **H1a, team block:** keep `team_win_pct` and `team_playoffs`; drop `team_srs`, `team_n_rtg` and `team_win_rank`.
- **H1b, scoring curvature:** replace `rank_pts_per_game` with `z_pts_per_game_sq` (the squared z-score). This is a new column computed in `features.build_table`, after the z-score loop.
  - If CV prefers the rank, the curvature reading is confirmed; the coefficient just looks odd.
- **H1c, regularisation:** add an H1c-only grid with C from 0.003 to 1 plus elastic net (`penalty="elasticnet", solver="saga", l1_ratio ∈ {0.2, 0.5}`).
  - **Keep the module-level `LR_GRID` frozen.** `run_all_nba` tunes every set with it, so widening it would re-tune A base too and leave H1c nothing to compare against.
  - Add an optional per-set `"grids"` key in `feature_sets` that `run_all_nba` passes to `T.tune`.

Files: `config.py` (feature lists), `features.py` (H1b column), `experiments.py` (per-set grid).

### H2. Too many centers in the positionless era

The training data cannot teach this (§2b), so it is handled by H6 below, using player archetypes rather than positions.

- **Now:** add the "Predicted C vs Actual C" table (§2b) to every evaluation report as a standing diagnostic.
- Do **not** add a `positionless` flag. It would be constant across the whole training set.

### H6. Player archetypes: centers count only if they drive the offense (user hypothesis, 2026-11-15)

**Rule to test:** a big should earn All-NBA credit for rebounds, blocks, defense and dunk-driven efficiency only if he is an offensive engine or shot creator. A lob/dunk finisher who also protects the rim (a "Rim-Running Big") should not.

The player types come from `../nba-archetype-pipeline` (k = 6, Kaggle source).

**Evidence (2014–2025, archetypes joined 1:1 on B-Ref `player_id`):**

| | Scoring Creator | Interior Scorer (Jokić, Embiid, AD, Sabonis, KAT, Wemby) | Rim-Running Big (Gobert, Zubac, Allen, Jordan) |
|---|---|---|---|
| All-NBA picks, positional ballots 2014–23 | 92 | 43 | **10** (8 on 3rd/2nd team) |
| All-NBA picks, positionless 2024–25 | 22 | 8 | **0** |
| MVP vote-getters 2014–25 (no position quota) | 97 | 44 | **3** |
| Model top 30, 2020–25: selection rate | 55% | 53% | **16%** |
| Model top 30, 2020–25: mean defense contribution | +0.46 | +1.26 | **+1.90** |

Voters without a position quota, on MVP ballots and the 2024+ All-NBA ballots, almost never pick rim-runners. The model still gives them the largest defensive bonus of any archetype.

**The catch:** the 1980–2023 positional ballots *did* select rim-runners to fill center slots (Jordan 1st team 2016, Gobert 2nd team 2017). A model trained on those seasons learns the opposite of the rule. CV on 2005–2019 is therefore expected to reject a learned version, for a reason that no longer applies.

**Making archetypes available for training:**
- Archetype labels exist from 2013-14 only.
- Every input the Kaggle archetype model uses (Advanced, Shooting, Play-by-Play, Career Info) exists from 1996-97.
- So build the archetype pipeline's season-relative features for 1997–2013 and apply the **frozen** model (`models/kaggle/archetype_model.joblib`: scaler, PCA, k-means). Do not refit.
- Check the extension by re-assigning 2014+ seasons through the same path: labels should agree with `player_archetypes.csv` at ≥ 95%.
- Spot-check 1997–2013 bigs (Shaq, Duncan, Ben Wallace, Tyson Chandler, Dwight Howard) **without** tuning anything.

**Archetype noise:** Ben Simmons 2020 is labelled Rim-Running Big. Apply the rule only when the archetype is Rim-Running Big **and** B-Ref `pos` is PF or C.

**Two versions:**
- **H6a, learned (eligible for adoption; 1997+ window, window reference B base):**
  - Add the archetype one-hots.
  - Add the interactions `rim_runner × {z_trb_per_game, z_blk_per_game, z_dws, z_ts_percent}`.
  - Also add the continuous self-creation inputs behind the archetype: `1 − percent_assisted_x2p_fg`, `percent_dunks_of_fga` and `z_usg_percent`, so the model is not limited to a hard label.
  - Expect CV to show little or no gain, because of the quota catch above. Report it anyway.
- **H6c, pre-registered rule for positionless seasons (not CV-adoptable):**
  - **As implemented:** in seasons ≥ `POSITIONLESS_FROM`, a *finisher big* has `z_trb_per_game`, `z_blk_per_game`, `z_dws` and `z_ts_percent` capped at 0 (league average) before scoring. This works with any model type. A finisher big is a Rim-Running Big, listed PF or C, who is **not his team's leading scorer**.
  - **Why the leading-scorer clause:** applying the frozen archetype model back to 1997 labels prime Shaq (2000 MVP) and prime Howard as Rim-Running Bigs. The archetype describes shot style, not offensive role. Exempting the team's top scorer keeps those offensive engines, and still catches Gobert, Zubac, Allen, Jordan and Duren.
  - The rule has no tuned parameter; the cap of 0 is fixed in advance from the MVP-ballot evidence.
  - It is designed after seeing 2024–25 (Gobert, Zubac), so those seasons cannot validate it. It is scored once on the 2026 labels, next to v1 and the adopted v3 set.
  - Sabonis is an Interior Scorer, so the rule leaves him alone. His 2025 false pick has a different cause: a .488 team, plus history.

Files (implemented):
- `../nba-archetype-pipeline/apply_frozen.py`: reproduces 100% of the 4,669 fitted 2014+ labels.
- `features.py`: `v3_features`, `finisher_big_rule`.
- `config.V3_GROUPS["H6a_archetype"]`.

### H7. Weak defense should not sink an offensive ace (user hypothesis, 2026-11-15)

**Evidence:** defense rarely pushes a creator *down*.
- Among top-30 Scoring Creators in 2020–25, 22 of 132 had a net negative defense contribution.
- The worst were Booker 2025 (−0.81), Lillard 2023 (−0.74), Lillard 2020 (−0.72) and Trae Young 2020 (−0.65).
- The larger effect is relative: bigs collect +1.3 to +1.9 from defense (H6). H7 is therefore expected to matter less than H6.

**Features (A window, 1980+; needs no archetype, so it can learn from the full history):**
- `def_z`: the mean of `z_dws`, `z_dbpm`, `z_stl_per_game` and `z_blk_per_game`.
- `offensive_ace`: 1 if the player leads his team in usage among rotation players **and** `z_pts_per_game ≥ 1`.
- Split defense into `def_pos = max(def_z, 0)` and `def_neg = min(def_z, 0)`, and add `def_neg × offensive_ace`.
- The model can then learn that negative defense costs an ace less. Keep the four raw defensive z-features, so the set changes only how negative defense is treated.

Files: `features.py` (team usage rank from the stint rows, like `team_vorp_share`), `config.FEATURE_GROUPS["defense_gate"]`.

### H8. Leading a much-improved team to a top record (user hypothesis, 2026-11-15)

**The pattern:** Ja Morant in 2022. Memphis went from .528 to .683 (2nd-best record) with Ja as its leading scorer; he made 2nd team, and v1 ranked him 17th.

**Features (A window):**
- `team_scoring_leader`.
- `lead_improver` = `team_scoring_leader × max(team_win_improvement, 0) × (team_win_rank ≤ 8)`. League rank 8 stands in for a top-3/4 conference seed, because seeds only exist from 1997.

Files: `features.v3_features`, `config.V3_GROUPS["H8_improver"]`.

### H3. Team-role features (rescoped from the old Improvements 1 and 4)

Binary "ace" flags add little: there are about 30 team top-scorers per season, most of them on bad teams. The `narrative` group (`team_best_share`, `team_vorp_share`) has already tested null. Use continuous features and a rank-based contender definition instead:

- **`team_pts_share`:** the player's points divided by the team's rotation-player points. For traded players, minutes-weight it across stints, like the other team features.
  - Stint rows currently carry only Advanced columns. Merge `pts_per_game × g` from the per-team rows of `Player Per Game.csv` onto `stints` before `teammate_features`.
- **`top2_on_contender`:** 1 if the player ranks first or second on his team by `team_pts_share` **and** `team_win_rank ≤ 4` within that season.
  - In 2025 that covers OKC, CLE, BOS and HOU. It replaces the fixed 0.610 threshold.

Honest expectation:
- Mitchell gains from `team_pts_share`; Jalen Williams gains only from `top2_on_contender`.
- Neither feature offsets the rebound, block and DWS gap that decided most of §2a.

### H4. Seeding (demoted)

`conf_seed` and `top2_seed` come from nba_api and only exist from 1997. The only information they add beyond `team_win_rank` is the conference split. Test them one more time:
- **The window reference** is `B base` (1997+), and it is a diagnostic.
- **The adoption bar** is unchanged: the gain over **A base** (1980+ v1) must have a CI above 0. `run_all_nba` already applies this rule to B sets.
- v2's in-window result was borderline harmful, so expect nothing.

### H5. Alternative efficiency ranks (low priority)

`z_ts_percent` is already a feature (coefficient +0.23). `rank_ts_percent` would mostly duplicate it and add the kind of collinearity H1 is meant to remove. Run it only if H1b shows the model benefits from rank-style terms.

### Not pursued

- Binary `ace_any` / `ace_majority` flags. The narrative group already tested null.
- A fixed win% contender threshold.
- Dropping `team_win_rank` outside the H1a block.

## 4. Experiment design (v3)

**Pre-register the sets before running anything.** Each one is compared to its base on its own. A cumulative chain (A → A+B → A+B+C) cannot attribute a gain to any one step.

| Set | Window | Adoption bar (CI > 0 vs) | Window reference (diagnostic) | Contents |
|---|---|---|---|---|
| v3-H1a | 1980+ | A base | A base | Team block reduced |
| v3-H1b | 1980+ | A base | A base | `rank_pts` replaced by `z_pts_per_game_sq` |
| v3-H1c | 1980+ | A base | A base | v1 features, H1c-only grid |
| v3-H3 | 1980+ | A base | A base | Adds `team_pts_share` and `top2_on_contender` |
| v3-H4 | 1997+ | A base | B base | Adds `conf_seed` and `top2_seed` |
| v3-H6a | 1997+ | A base | B base | Archetype one-hots, rim-runner interactions, self-creation shooting columns |
| v3-H7 | 1980+ | A base | A base | `def_pos`, `def_neg`, `def_neg × offensive_ace` |
| v3-H8 | 1980+ | A base | A base | `team_scoring_leader`, `lead_improver` |
| v3-combo | 1980+ / 1997+ | A base | — | All sets that passed. Run afterwards as confirmation. |
| v3-H6c rule | — | not eligible | — | Rim-runner cap for 2024+ seasons. Scored only on the 2026 labels. |

**Rules (the v2 harness):**
- Forward-chaining CV on `V2_CV_BLOCKS` (2005–2019, 15 seasons).
- The metric is the mean per-season PR-AUC gain, with a paired bootstrap over seasons.
- **Bonferroni count: m = 16.** That is 8 eligible sets (H1a, H1b, H1c, H3, H4, H6a, H7, H8) × 2 passes.
  - **Pass 1 (v2 harness):** each set takes whichever of logreg and hist_gb wins CV.
  - **Pass 2 (`reports/v3_logreg/`):** every set and base is logistic regression.
  - Pass 2 was added after pass 1, because pass 1 mixes the feature effect with a change of model type (A base is logreg, most sets chose hist_gb). Counting both passes in m keeps that second look honest.
- Parsimony: among adopted sets within 0.002 CV PR-AUC of each other, take the one with fewer features.
- Every candidate uses the same Platt calibration and the same 65-game eligibility mask.

**What to expect:**
- v2's CIs were about ±0.02–0.03 PR-AUC wide, and none of its 16 sets cleared them.
- A change that moves two players in one season is far below what 15 CV seasons can detect.
- **"Nothing adopted" is a likely and legitimate outcome.** In that case keep v1 and record the null result, as v2 did.

### Holdout and test policy

- **2020–2025 holdout: report it, never choose with it.** It is now contaminated, because H1–H3 were shaped by looking at 2020–2025 misses. Holdout hits for v3 sets will be optimistic.
- **Clean test: 2025–26 (`season = 2026`).**
  - The local Kaggle copy has All-NBA labels only through 2025 (`build_info.json`).
  - The 2025–26 teams have probably been announced by now. `reports/predictions_2026.csv` already exists, but it has not been compared with them.
  - **Freeze the v3 choice first**, then download the 2026 labels and score the adopted set and v1 **once**.
  - Do not look at 2026 misses before the choice is frozen.
- **Diagnostics, not success criteria:**
  - The 2025 rank of Mitchell and J. Williams.
  - How many of the 16 holdout misses are recovered.
  - The predicted-vs-actual center table (§2b).
  - Do not track the two named players alone: that would reward features that overfit to them.

### Success criteria

1. At least one eligible v3 set has a CV gain over A base whose CI lower bound is above 0. Otherwise v1 stays.
2. The adopted set does not lose holdout hits against v1 (73/90). This is a check, not a selector.
3. On the 2026 labels, the adopted set scores at least as well as v1.

## 5. Implementation order

| Step | Work | Effort | Why this order |
|---|---|---|---|
| 1 | Write the v3 `feature_sets` list, add the per-set `"grids"` key and the `n_tests = 7` override in `experiments.py` | Low | Locks the comparisons before any results are seen |
| 2 | H1a and H1c (config and grid only), and H1b (`z_pts_per_game_sq` in `features.py`) | Low | Cheap test of the collinearity hypothesis |
| 3 | Extend archetypes to 1997 with the frozen model (archetype pipeline), then run the ≥ 95% agreement check | Medium | Needed by H6. Do it before any feature work that depends on it. |
| 4 | H6a and H7 features in `features.py`; write the H6c rule as a scoring post-step in `predict.py` | Medium | Directly addresses two of the hypotheses |
| 5 | H3: `team_pts_share` and `top2_on_contender`, using per-stint points | Medium | Plausible, but weaker evidence |
| 6 | H4 (1997+ window), then the combo set | Low | H4's prior evidence is negative |
| 7 | Write `reports/v3_report.md` with the same tables as v2, plus the §4 diagnostics and the §2b center table | Low | |
| 8 | Freeze the choice and the H6c rule, pull the 2026 labels, run the one-shot test | Low | The only unbiased read on v3 and on H6c |

Add tests to `tests_smoke.py` checking that:
- `team_pts_share` sums to at most 1 per team-season.
- `z_pts_per_game_sq` equals `z_pts_per_game ** 2`.
- Archetypes from the frozen model, re-run on 2014+, agree with `player_archetypes.csv` on at least 95% of player-seasons.
- `def_pos + def_neg == def_z` on every row.
- No v3 feature uses same-season labels.

## 6. Bottom line

The 2024–25 misses were two players just below the cutoff. `team_best_share` did not cause them, because the model does not use it. `team_win_rank` did not cause them either; it raised both players' scores.

The gap came from rebound, block and DWS terms. Two verified problems are worth testing:
- Collinear team and scoring features with sign-flipped coefficients (H1).
- A center surplus that appeared only after the positionless ballot (H2). Player archetypes point to the cause: quota-free voters (MVP ballots, 2024+ All-NBA) almost never pick rim-running bigs, yet the model gives them the biggest defensive bonus. The fix is H6, a learned version plus a pre-registered rule for positionless seasons, together with H7 so weak defense matters less for offensive aces.

Feature selection uses CV only, and 2025–26 is the clean test. Given v2's history, expect small or null gains, and treat a null result as an answer rather than a failure.

## 7. v3 results (run 2026-11-15)

Full tables: `reports/v3_report.md` (pass 1), `reports/v3_logreg/v3_report.md` (pass 2), `reports/v3_candidate.md`.

**No set was adopted. The served model is still v1** (`models/all_nba_v3.joblib` holds the v1 bundle).

CV gain vs the v1 features (2005–2019, per-season PR-AUC), from the logistic-only pass, which is the clean feature test:

| Set | Gain | Bonferroni CI (m = 16) | Verdict |
|---|---|---|---|
| H1a team block | −0.0036 | [−0.010, +0.002] | null |
| H1b pts curvature | −0.0043 | [−0.017, +0.006] | null |
| H1c wider regularisation | −0.0001 | [−0.002, +0.001] | null (C = 0.3 with elastic net ≈ v1) |
| H3 team role | −0.0056 | [−0.011, −0.0004] | **reliably worse** |
| H4 seed (vs B base) | −0.0015 | [−0.006, +0.002] | null |
| H6a archetype (vs B base) | +0.0055 | [−0.007, +0.019] | null; widest upside |
| H7 defense gate | −0.0041 | [−0.009, +0.001] | null |
| H8 improver | +0.0004 | [−0.012, +0.010] | null |

**What this means:**
- On 2005–2019, under positional ballots, none of these ideas changes how well the model ranks players.
- That does not make the ideas wrong. H6's center rule cannot show up in those years, because of the quota (§H6).
- The two that target the newest voting behaviour, the center rule and the improver, can only be judged on positionless seasons.

**Opt-in candidate:**
- **Definition:** three hypotheses, fixed before testing. It is the v1 features plus H7 and H8 (logreg, C = 0.1), with the H6c rule applied at scoring for seasons from 2024. The spec is saved inside `models/all_nba_v3_candidate.joblib`.
- **CV gain:** −0.0010, CI [−0.010, +0.007].
- **Holdout:** 76/90 against v1's 74/90 (+1 in each of 2020 and 2022, Ja Morant among them). Those seasons shaped the ideas, so this is optimistic.
- **How to see it:** `predict.py --candidate` prints it beside v1.
- **For 2025-26** it differs from v1 by one player: Chet Holmgren in, Jalen Duren (a finisher big) out.

**Clean test:**
- A fresh Kaggle download (2026-11-15) still has All-NBA labels only through 2024-25.
- When the 2025-26 labels are available, score v1 and the candidate once on `predictions_2026.csv`. Switch the default only if the candidate wins.
- ±1 hit out of 15 is noise, so one season can only flag a large difference. A fair verdict needs 2026 and 2027 together.
