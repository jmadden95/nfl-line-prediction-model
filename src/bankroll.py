"""Bankroll over time = unstaked funds + stakes in open bets (ported from nrl-line).

SHARED with nrl-line via src/shared_bankroll.py (ledger at /app/data/shared): one
balance across both apps, each app's bets and performance tracked separately.

Anchor your CURRENT unstaked balance once; thereafter the total moves only by
realised P/L from settled bets. Re-anchor after a deposit/withdrawal.
    total    = anchor_total + (realised_now - realised_at_anchor)
    pending  = stakes in unsettled bets
    unstaked = total - pending

    python -m src.bankroll set 200     # record your unstaked balance
    python -m src.bankroll             # snapshot today + print
"""
import sys

import pandas as pd

from config import LOCAL_TZ
from src import db


def _state():
    from src.bets import settle_bets
    bets = settle_bets(db.read_bets())
    if bets is None or not len(bets):
        return 0.0, 0.0, 0
    settled = bets["result"].isin(["W", "L", "P"])
    pending = float(pd.to_numeric(bets.loc[~settled, "stake"], errors="coerce").fillna(0).sum())
    realised = float(pd.to_numeric(bets.loc[settled, "profit"], errors="coerce").fillna(0).sum())
    return pending, realised, int((~settled).sum())


APP = "nfl"


def _shared() -> bool:
    try:
        from src import shared_bankroll as sh
        return sh.available()
    except Exception:  # noqa: BLE001
        return False


def _publish():
    from src import shared_bankroll as sh
    pending, realised, npend = _state()
    sh.publish(APP, realised, pending, npend)
    return pending, realised, npend


def _from_shared(c: dict) -> dict:
    own = (c.get("apps") or {}).get(APP, {})
    return {"total": c["total"], "pending": c["pending"], "unstaked": c["unstaked"],
            "n_pending": c["n_pending"], "anchor_date": c["anchor_date"],
            "since_anchor": c["since_anchor"], "own_since_anchor": own.get("since_anchor", 0.0),
            "apps": c.get("apps", {}), "shared": True}


def set_anchor(unstaked: float) -> dict:
    if _shared():
        from src import shared_bankroll as sh
        _publish()
        sh.set_anchor(float(unstaked))
        return snapshot()
    pending, realised, _ = _state()
    db.set_kv("bankroll_anchor_date", pd.Timestamp.now(tz=LOCAL_TZ).date().isoformat())
    db.set_kv("bankroll_anchor_total", float(unstaked) + pending)
    db.set_kv("bankroll_anchor_realised", realised)
    return snapshot()


def current():
    if _shared():
        from src import shared_bankroll as sh
        _publish()
        c = sh.current()
        return _from_shared(c) if c else None
    tot = db.get_kv("bankroll_anchor_total")
    if tot is None:
        return None
    pending, realised, npend = _state()
    total = float(tot) + realised - float(db.get_kv("bankroll_anchor_realised", 0) or 0)
    return {"total": total, "pending": pending, "unstaked": total - pending, "n_pending": npend,
            "anchor_date": db.get_kv("bankroll_anchor_date"),
            "since_anchor": realised - float(db.get_kv("bankroll_anchor_realised", 0) or 0)}


def snapshot() -> dict | None:
    cur = current()
    if cur is None:
        return None
    if cur.get("shared"):
        from src import shared_bankroll as sh
        sh.snapshot()
        return cur
    conn = db.connect()
    try:
        conn.execute("INSERT OR REPLACE INTO bankroll_history VALUES (?,?,?,?,?)",
                     (pd.Timestamp.now(tz=LOCAL_TZ).date().isoformat(), cur["total"], cur["unstaked"],
                      cur["pending"], cur["since_anchor"]))
        conn.commit()
    finally:
        conn.close()
    return cur


def history() -> pd.DataFrame:
    if _shared():
        from src import shared_bankroll as sh
        return sh.history()
    return db._df("SELECT * FROM bankroll_history ORDER BY date")


def kelly_fraction() -> float:
    try:
        return float(db.get_kv("kelly_fraction", 0.25))
    except Exception:  # noqa: BLE001
        return 0.25


def kelly_stake(model_ev, odds, bank_total, fraction=None, inc=1):
    """Fractional-Kelly stake (units), rounded to `inc`; None if no positive edge."""
    try:
        fraction = kelly_fraction() if fraction is None else fraction
        if model_ev is None or model_ev <= 0 or not bank_total:
            return None
        b = float(odds) - 1.0
        if b <= 0:
            return None
        stake = fraction * (model_ev / b) * float(bank_total)
        r = round(stake / inc) * inc
        return r if r > 0 else None
    except Exception:  # noqa: BLE001
        return None


if __name__ == "__main__":
    if len(sys.argv) >= 3 and sys.argv[1] == "set":
        print(set_anchor(float(sys.argv[2])))
    else:
        print(snapshot() or "no bankroll anchored yet (python -m src.bankroll set <unstaked>)")
        print(history().tail(10).to_string(index=False))
