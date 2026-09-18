"""Predicted margin -> win probability (Normal with fixed sd, plus a Platt fit)."""
import numpy as np
from scipy.stats import norm
from sklearn.linear_model import LogisticRegression

from config import MARGIN_STD_DEV


def margin_to_win_prob(pred_margin, sd: float = MARGIN_STD_DEV):
    return norm.cdf(np.asarray(pred_margin, dtype=float) / sd)


def cover_prob(edge, sd: float):
    """P(the side with `edge` points of model value covers) — sd is the betting
    sigma (wider than the raw margin sd, since a flagged edge is partly noise)."""
    return norm.cdf(np.asarray(edge, dtype=float) / sd)


def fit_winprob_calibrator(margins, outcomes) -> LogisticRegression:
    X = np.asarray(margins, dtype=float).reshape(-1, 1)
    return LogisticRegression().fit(X, np.asarray(outcomes, dtype=int))


def logistic_win_prob(cal, margins):
    return cal.predict_proba(np.asarray(margins, dtype=float).reshape(-1, 1))[:, 1]
