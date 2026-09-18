"""ONE bankroll shared by every betting app (nrl-line, nfl-line), performance kept per app.

A tiny SQLite ledger on a host directory bind-mounted into each container at
/app/data/shared. Each app PUBLISHES only its own numbers — realised P/L from
its settled bets and the stakes tied up in its open bets — and the ledger
combines them:

    total    = anchor_total + Σ_app (realised_app − realised_app_at_anchor)
    pending  = Σ_app pending_app
    unstaked = total − pending

Anchor once with your real unstaked balance (from either app); re-anchor after
a deposit/withdrawal. Apps that didn't exist at anchor time count from zero.
If the shared path isn't mounted, callers fall back to their local bankroll.
"""
from __future__ import annotations

import os
import sqlite3
from pathlib import Path

import pandas as pd

SHARED_DB = Path(os.getenv("SHARED_BANKROLL_DB", "/app/data/shared/bankroll.db"))


def available() -> bool:
    return SHARED_DB.parent.is_dir()


def _connect() -> sqlite3.Connection:
    conn = sqlite3.connect(SHARED_DB, timeout=15, check_same_thread=False)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA busy_timeout=8000")
    conn.execute("CREATE TABLE IF NOT EXISTS anchor(key TEXT PRIMARY KEY, value TEXT)")
    conn.execute("CREATE TABLE IF NOT EXISTS app_state(app TEXT PRIMARY KEY, realised REAL, "
                 "pending REAL, n_pending INTEGER, updated TEXT)")
    conn.execute("CREATE TABLE IF NOT EXISTS history(date TEXT PRIMARY KEY, total REAL, "
                 "unstaked REAL, pending REAL)")
    return conn


def _get(conn, key, default=None):
    r = conn.execute("SELECT value FROM anchor WHERE key=?", (key,)).fetchone()
    return r[0] if r else default


def _set(conn, key, value):
    conn.execute("INSERT INTO anchor(key,value) VALUES(?,?) ON CONFLICT(key) DO UPDATE SET value=excluded.value",
                 (key, str(value)))


def _today() -> str:
    return str(pd.Timestamp.now().date())


def publish(app: str, realised: float, pending: float, n_pending: int) -> None:
    conn = _connect()
    try:
        conn.execute("INSERT INTO app_state VALUES(?,?,?,?,?) ON CONFLICT(app) DO UPDATE SET "
                     "realised=excluded.realised, pending=excluded.pending, n_pending=excluded.n_pending, "
                     "updated=excluded.updated",
                     (app, float(realised), float(pending), int(n_pending), pd.Timestamp.now().isoformat(timespec="seconds")))
        conn.commit()
    finally:
        conn.close()


def _apps(conn) -> dict:
    return {r[0]: {"realised": float(r[1] or 0), "pending": float(r[2] or 0), "n_pending": int(r[3] or 0)}
            for r in conn.execute("SELECT app, realised, pending, n_pending FROM app_state")}


def has_anchor() -> bool:
    conn = _connect()
    try:
        return _get(conn, "anchor_total") is not None
    finally:
        conn.close()


def set_anchor(unstaked: float, when: str = None) -> dict:
    """Anchor to the real unstaked balance NOW (all apps' pending stakes added on top)."""
    conn = _connect()
    try:
        apps = _apps(conn)
        _set(conn, "anchor_date", when or _today())
        _set(conn, "anchor_total", float(unstaked) + sum(a["pending"] for a in apps.values()))
        for name, a in apps.items():
            _set(conn, f"anchor_realised_{name}", a["realised"])
        conn.commit()
    finally:
        conn.close()
    return current()


def seed_anchor(date: str, total: float, realised_by_app: dict) -> None:
    """Migrate an app's pre-existing local anchor into the ledger (first run only)."""
    conn = _connect()
    try:
        if _get(conn, "anchor_total") is not None:
            return
        _set(conn, "anchor_date", date)
        _set(conn, "anchor_total", float(total))
        for name, r in realised_by_app.items():
            _set(conn, f"anchor_realised_{name}", float(r))
        conn.commit()
    finally:
        conn.close()


def current() -> dict | None:
    conn = _connect()
    try:
        at = _get(conn, "anchor_total")
        if at is None:
            return None
        apps = _apps(conn)
        per_app = {}
        for name, a in apps.items():
            since = a["realised"] - float(_get(conn, f"anchor_realised_{name}", 0) or 0)
            per_app[name] = {"since_anchor": since, "pending": a["pending"], "n_pending": a["n_pending"]}
        since_total = sum(v["since_anchor"] for v in per_app.values())
        pending = sum(v["pending"] for v in per_app.values())
        total = float(at) + since_total
        return {"date": _today(), "anchor_date": _get(conn, "anchor_date"), "total": total,
                "pending": pending, "unstaked": total - pending,
                "n_pending": sum(v["n_pending"] for v in per_app.values()),
                "since_anchor": since_total, "apps": per_app}
    finally:
        conn.close()


def snapshot() -> dict | None:
    cur = current()
    if not cur:
        return None
    conn = _connect()
    try:
        conn.execute("INSERT INTO history(date,total,unstaked,pending) VALUES(?,?,?,?) ON CONFLICT(date) DO UPDATE "
                     "SET total=excluded.total, unstaked=excluded.unstaked, pending=excluded.pending",
                     (cur["date"], cur["total"], cur["unstaked"], cur["pending"]))
        conn.commit()
    finally:
        conn.close()
    return cur


def history() -> pd.DataFrame:
    conn = _connect()
    try:
        return pd.read_sql("SELECT * FROM history ORDER BY date", conn)
    finally:
        conn.close()
