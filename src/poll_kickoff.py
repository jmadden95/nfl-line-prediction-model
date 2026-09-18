"""Capture each game's line ~10 min before kickoff (the close, for CLV).

Cheap: kickoff times come from the cached odds (free); the API is only hit
when a game is inside the window AND not yet captured. Cron every 5 min across
the game windows (see deploy/nfl-line.cron).
"""
import pandas as pd

from config import LOCAL_TZ
from src import db
from src.live import cached_events, fetch_odds

WINDOW_MIN = 10


def _captured(home, away, gd) -> bool:
    conn = db.connect()
    try:
        return conn.execute("SELECT 1 FROM odds_snapshots WHERE phase='kickoff' AND game_date=? "
                            "AND home=? AND away=? LIMIT 1", (gd, home, away)).fetchone() is not None
    finally:
        conn.close()


def main() -> None:
    db.init_db()
    now = pd.Timestamp.now(tz=LOCAL_TZ).tz_localize(None)
    cached = cached_events()
    if cached:
        due = [e for e in cached if 0 < (e["kickoff"] - now).total_seconds() / 60 <= WINDOW_MIN
               and not _captured(e["home"], e["away"], str(e["date"].date()))]
        if not due:
            print("poll_kickoff: nothing due — skipped (no API call).")
            return
    try:
        events = fetch_odds(use_cache=False)
    except Exception as ex:  # noqa: BLE001
        print(f"poll_kickoff: odds fetch failed ({ex})")
        return
    n = 0
    for e in events:
        if 0 < (e["kickoff"] - now).total_seconds() / 60 <= WINDOW_MIN:
            n += db.snapshot_game(e, phase="kickoff")
    db.snapshot_book_lines(events, phase="kickoff")
    print(f"poll_kickoff: captured {n} game(s) at the close")
    try:
        from src.notify import alert_line_moves, alert_sharp_lags
        alert_sharp_lags(events); alert_line_moves()
    except Exception as ex:  # noqa: BLE001
        print(f"poll_kickoff: alerts failed ({ex})")


if __name__ == "__main__":
    main()
