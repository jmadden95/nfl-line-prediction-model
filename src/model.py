"""Baseline: linear regression on leakage-free features vs the two honest baselines.

    python -m src.model
"""
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error

from config import TEST_SEASONS, TRAIN_FROM_SEASON
from src.features import FEATURE_COLUMNS, LINE_FREE_COLUMNS, build_features
from src.load_data import add_margin, load_raw


def load_modelling_frame() -> pd.DataFrame:
    df = build_features(add_margin(load_raw()))
    df = df[df["season"] >= TRAIN_FROM_SEASON].copy()
    df = df.dropna(subset=["home_margin", "bookie_margin"])
    df[FEATURE_COLUMNS] = df[FEATURE_COLUMNS].fillna(0.0)
    return df


def split_train_test(df):
    test_start = min(TEST_SEASONS)
    return df[df["season"] < test_start], df[df["season"].isin(TEST_SEASONS)]


def main() -> None:
    df = load_modelling_frame()
    train, test = split_train_test(df)
    print(f"Train {len(train)} games ({train['season'].min()}-{train['season'].max()}) | "
          f"Test {len(test)} games (season {TEST_SEASONS})")
    y = test["home_margin"]
    avg = train["home_margin"].mean()
    print(f"\n{'model':<34}{'MAE':>8}")
    print(f"{'(a) always avg (%+.1f)' % avg:<34}{mean_absolute_error(y, np.full(len(test), avg)):>8.2f}")
    print(f"{'(b) AU opening line':<34}{mean_absolute_error(y, test['bookie_margin']):>8.2f}")
    if test["close_margin"].notna().all():
        print(f"{'(b2) AU closing line':<34}{mean_absolute_error(y, test['close_margin']):>8.2f}")
    full = LinearRegression().fit(train[FEATURE_COLUMNS], train["home_margin"])
    print(f"{'linear (with line)':<34}{mean_absolute_error(y, full.predict(test[FEATURE_COLUMNS])):>8.2f}")
    nl = LinearRegression().fit(train[LINE_FREE_COLUMNS], train["home_margin"])
    p = nl.predict(test[LINE_FREE_COLUMNS])
    print(f"{'linear LINE-FREE':<34}{mean_absolute_error(y, p):>8.2f}")
    print(f"{'50/50 blend line-free + line':<34}{mean_absolute_error(y, 0.5 * p + 0.5 * test['bookie_margin']):>8.2f}")
    print("\nLine-free coefficients (pts of home margin per unit):")
    for name, c in sorted(zip(LINE_FREE_COLUMNS, nl.coef_), key=lambda x: -abs(x[1])):
        print(f"  {name:20s} {c:+.3f}")
    print(f"  {'(intercept = HFA)':20s} {nl.intercept_:+.3f}")


if __name__ == "__main__":
    main()
