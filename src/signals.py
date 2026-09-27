"""One place for the bet-signal rules (board, alerts, scorecards, backtest).

Validated walk-forward, rule chosen on 2012-18 and checked on untouched 2019-25:
  HANDICAP  back the side the model favours vs the AU line when the edge is
            >= FAV_EDGE on the line's FAVOURITE, or >= DOG_EDGE on the UNDERDOG.
            Why asymmetric: a line-free regression shrinks every prediction
            toward zero, so vs a big market favourite it manufactures fake
            'value on the dog' (64% of the old |edge|>=2 signals were dogs, and
            those were break-even). Train +0.2% -> +5.6% ROI; validate +26.6%.
  TOTALS    (model has wind) UNDER when model <= AU total - TOTAL_UNDER_EDGE,
            OVER when model >= AU total + TOTAL_OVER_EDGE. Unders at 2+ were
            +7.8% / +8.7% (train/validate); overs only pay at 3+.
"""
from config import DOG_EDGE, FAV_EDGE, TOTAL_OVER_EDGE, TOTAL_UNDER_EDGE


def handicap_signal(edge, bookie_margin):
    """edge = blended model margin - AU line margin (home view); bookie_margin =
    -AU home point. -> 'home' | 'away' | None."""
    try:
        e, bm = float(edge), float(bookie_margin)
    except (TypeError, ValueError):
        return None
    if e != e or bm != bm or e == 0:
        return None
    backs_fav = (e > 0) == (bm > 0) and bm != 0
    need = FAV_EDGE if backs_fav else DOG_EDGE
    if abs(e) < need:
        return None
    return "home" if e > 0 else "away"


def total_signal(t_edge):
    """t_edge = model total - AU total -> 'over' | 'under' | None."""
    try:
        t = float(t_edge)
    except (TypeError, ValueError):
        return None
    if t != t:
        return None
    if t <= -TOTAL_UNDER_EDGE:
        return "under"
    if t >= TOTAL_OVER_EDGE:
        return "over"
    return None
