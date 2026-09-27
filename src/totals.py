"""Totals (over/under): a simple leakage-free model blended toward the AU total."""
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

from config import MARKET_BLEND, TRAIN_FROM_SEASON

# wind_f: mph at kickoff, 0 when enclosed — walk-forward ~ -0.26 pts/mph (src/weather.py)
TOTAL_FEATURES = ["exp_total", "is_dome", "div_game", "is_playoff", "wind_f"]


def _exp_total(df: pd.DataFrame) -> pd.Series:
    return ((df["home_attack"] + df["away_defense"]) / 2.0 + (df["away_attack"] + df["home_defense"]) / 2.0)


def predict_totals(feat: pd.DataFrame, up: pd.DataFrame, au_total) -> np.ndarray:
    """feat = featured history (completed), up = featured upcoming rows."""
    hist = feat[(feat["season"] >= TRAIN_FROM_SEASON)].dropna(subset=["total_points"]).copy()
    hist["exp_total"] = _exp_total(hist)
    hist = hist.dropna(subset=["exp_total"])
    hist[TOTAL_FEATURES] = hist[TOTAL_FEATURES].fillna(0.0)
    # recency weight: scoring environment drifts (rule changes); recent seasons count more
    w = 0.85 ** (hist["season"].max() - hist["season"])
    m = LinearRegression().fit(hist[TOTAL_FEATURES], hist["total_points"], sample_weight=w)
    u = up.copy()
    u["exp_total"] = _exp_total(u).fillna(hist["exp_total"].mean())
    u[TOTAL_FEATURES] = u[TOTAL_FEATURES].fillna(0.0)
    pure = m.predict(u[TOTAL_FEATURES])
    au = pd.to_numeric(pd.Series(au_total), errors="coerce").to_numpy(dtype=float)
    return np.where(np.isnan(au), pure, (1 - MARKET_BLEND) * pure + MARKET_BLEND * au)
