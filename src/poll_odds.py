"""Scheduled line capture — AU line + sharp line at key points of the NFL week.

  open      Tue morning (AU): the look-ahead line, before the practice reports
  mid       Fri morning (AU): after Wed/Thu practice reports
  late      Sat morning (AU): after Friday's final injury report
  pre_kick  Sun/Mon morning (AU): shortly before the Sunday slate

    python -m src.poll_odds --phase open
"""
import argparse

from src import db
from src.live import fetch_odds


def run(phase: str) -> None:
    db.init_db()
    try:
        events = fetch_odds(use_cache=False)
    except Exception as ex:  # noqa: BLE001
        print(f"poll {phase}: odds fetch failed ({ex})")
        return
    n = db.snapshot_odds(events, phase=phase)
    db.snapshot_book_lines(events, phase=phase)
    print(f"poll {phase}: captured {n} games" + ("" if n else " (already captured today)"))
    try:
        from src.notify import alert_line_moves, alert_sharp_lags
        print(f"poll {phase}: {alert_sharp_lags(events)} sharp-lag alert(s), "
              f"{alert_line_moves()} line-move alert(s)")
    except Exception as ex:  # noqa: BLE001
        print(f"poll {phase}: alert check failed ({ex})")
    try:
        from src.live import predict_upcoming
        from src.notify import alert_bet_signals
        up = predict_upcoming(events)
        db.snapshot_predictions(up, phase=phase)
        print(f"poll {phase}: {alert_bet_signals(up)} bet-signal alert(s)")
    except Exception as ex:  # noqa: BLE001
        print(f"poll {phase}: prediction/bet-signal failed ({ex})")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", required=True, choices=["open", "mid", "late", "pre_kick", "page"])
    run(ap.parse_args().phase)
