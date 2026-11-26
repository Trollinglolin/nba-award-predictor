"""Shared evaluation helpers: per-season top-k, eligibility mask, positional allocation, calibration."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score

import config as C

# Slots per position for position-based ballots (pre-2024): each team = 2 G, 2 F, 1 C.
SLOTS_PER_TEAM = {"G": 2, "F": 2, "C": 1}


def eligibility_games(df: pd.DataFrame, exact: bool = False) -> pd.Series:
    """Games that count toward the 65-game rule. exact=True uses nba_api game logs (20+ min games, plus up to two
    15-20 min games) where available and falls back to total games elsewhere."""
    if exact and "g_counted" in df.columns:
        return df["g_counted"].fillna(df["g"])
    return df["g"]


def apply_eligibility(df: pd.DataFrame, score: pd.Series, rule: bool = True, exact: bool = False) -> pd.Series:
    """2024+ 65-game rule. Default approximation: total games (necessary, not sufficient)."""
    if not rule:
        return score
    games = eligibility_games(df, exact)
    inelig = (df["season"] >= C.GAMES_RULE_FROM) & (games < C.GAMES_RULE_MIN_G * df["season_scale"])
    return score.astype(float).where(~inelig, -np.inf)


def top_k(df: pd.DataFrame, score: pd.Series, k_map: dict, positional: bool = False) -> pd.Series:
    """Boolean Series: True for the players picked in each season (top-k, optionally slot-constrained)."""
    picked = pd.Series(False, index=df.index)
    for season, g in df.assign(_s=score).groupby("season"):
        k = int(k_map.get(season, 15))
        order = g.sort_values("_s", ascending=False, kind="mergesort")
        if positional and season < C.POSITIONLESS_FROM:
            n_teams = k // 5
            left = {p: n * n_teams for p, n in SLOTS_PER_TEAM.items()}
            chosen = []
            for idx, pos in zip(order.index, order["pos_group"]):
                if left.get(pos, 0) > 0:
                    chosen.append(idx)
                    left[pos] -= 1
                if len(chosen) == k:
                    break
        else:
            chosen = list(order.index[:k])
        picked.loc[chosen] = True
    return picked


def per_season_hits(df, score, label_col, k_map, positional=False) -> pd.DataFrame:
    picked = top_k(df, score, k_map, positional)
    out = (df.assign(_p=picked, _y=df[label_col])
             .groupby("season").apply(lambda g: pd.Series({"k": k_map[g.name], "hits": int((g._p & (g._y == 1)).sum())}),
                                      include_groups=False))
    out["precision_at_k"] = out["hits"] / out["k"]
    return out.reset_index()


def classification_metrics(df, score, label_col, k_map, prob=True, positional=False) -> dict:
    y = df[label_col].astype(int)
    s = score.replace(-np.inf, np.nan).fillna(score.replace(-np.inf, np.nan).min() - 1)
    ps = per_season_hits(df, score, label_col, k_map, positional)
    out = {"pr_auc": float(average_precision_score(y, s)), "roc_auc": float(roc_auc_score(y, s)),
           "precision_at_k": float(ps["hits"].sum() / ps["k"].sum()), "hits": int(ps["hits"].sum()),
           "slots": int(ps["k"].sum())}
    if prob:
        out["brier"] = float(brier_score_loss(y, s.clip(0, 1)))
    return out


class PlattCalibrated:
    """Monotone (Platt) recalibration of a fitted classifier: q = sigmoid(a * logit(p) + b).

    a, b are fitted on out-of-fold forward-chaining predictions (each season scored by a model trained
    only on earlier seasons), so calibration never sees the holdout. Rankings are unchanged."""

    def __init__(self, base, a: float, b: float):
        self.base, self.a, self.b = base, a, b

    @staticmethod
    def _logit(p):
        p = np.clip(p, 1e-6, 1 - 1e-6)
        return np.log(p / (1 - p))

    @classmethod
    def fit_params(cls, p_oof, y_oof):
        from sklearn.linear_model import LogisticRegression
        lr = LogisticRegression(C=1e6, max_iter=1000).fit(cls._logit(np.asarray(p_oof)).reshape(-1, 1), np.asarray(y_oof))
        return float(lr.coef_[0, 0]), float(lr.intercept_[0])

    def predict_proba(self, X):
        z = self.a * self._logit(self.base.predict_proba(X)[:, 1]) + self.b
        q = 1 / (1 + np.exp(-z))
        return np.column_stack([1 - q, q])


def reliability_table(y, p, bins=(0, .02, .05, .1, .2, .4, .6, .8, 1.0001)) -> pd.DataFrame:
    d = pd.DataFrame({"y": np.asarray(y), "p": np.asarray(p)})
    d["bin"] = pd.cut(d["p"], bins, right=False)
    return d.groupby("bin", observed=True).agg(n=("y", "size"), mean_pred=("p", "mean"), observed=("y", "mean")).reset_index()


def team_assignment_accuracy(df, score, k_map) -> float:
    """Among true selections that were picked: share placed on the right team (rank 1-5 = 1st, 6-10 = 2nd, ...)."""
    right = total = 0
    for season, g in df.assign(_s=score).groupby("season"):
        k = int(k_map[season])
        top = g.sort_values("_s", ascending=False).head(k)
        pred_team = np.arange(len(top)) // 5 + 1
        true = top["all_nba_team"].to_numpy()
        mask = true > 0
        right += int((pred_team[mask] == true[mask]).sum())
        total += int(mask.sum())
    return right / total if total else float("nan")
