"""Single SQLite store: results, odds snapshots (AU + sharp), outs, bets, predictions.

data/nfl.db (bind-mounted, git-ignored). Mirrors the NRL store.

Tables
  matches            workbook rows (re-seeded when the xlsx changes)
  auto_matches       results auto-captured from ESPN, paired with captured lines
  player_outs        live outs (entered, weeks, team, player, position, points, kind, source)
  bets               bet log
  odds_snapshots     per game per phase per day: AU line + sharp line (+ totals, h2h)
  book_lines         EVERY book's line per capture (AU vs sharp lag analysis)
  predictions        model output captured at view/poll time
  model_coeffs, sent_alerts, api_usage, kv
"""
from __future__ import annotations

import datetime
import json
import sqlite3
from pathlib import Path

import pandas as pd

from config import DATA_DIR, RAW_XLSX, LOCAL_TZ

DB_PATH = DATA_DIR / "nfl.db"
OUT_COLUMNS = ["entered", "weeks", "team", "player", "position", "points", "kind", "source", "note"]
BET_COLUMNS = ["logged", "gdate", "home", "away", "side", "team", "line", "odds", "stake",
               "book", "home_score", "away_score", "close_line", "result", "profit", "model"]


def connect() -> sqlite3.Connection:
    DATA_DIR.mkdir(exist_ok=True)
    conn = sqlite3.connect(DB_PATH, check_same_thread=False, timeout=15)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA busy_timeout=8000")
    return conn


def _today() -> str:
    return pd.Timestamp.now(tz=LOCAL_TZ).date().isoformat()


def _now_iso() -> str:
    return pd.Timestamp.now(tz=LOCAL_TZ).strftime("%Y-%m-%dT%H:%M:%S")


def _has_table(conn, table) -> bool:
    return conn.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name=?",
                        (table,)).fetchone() is not None


def _has_rows(conn, table) -> bool:
    try:
        return conn.execute(f'SELECT 1 FROM "{table}" LIMIT 1').fetchone() is not None
    except sqlite3.OperationalError:
        return False


_SCHEMA = {
    "player_outs": "entered TEXT, weeks REAL, team TEXT, player TEXT, position TEXT, "
                   "points REAL, kind TEXT, source TEXT, note TEXT",
    "bets": "logged TEXT, gdate TEXT, home TEXT, away TEXT, side TEXT, team TEXT, line REAL, "
            "odds REAL, stake REAL, book TEXT, home_score REAL, away_score REAL, "
            "close_line REAL, result TEXT, profit REAL, model TEXT",
    "odds_snapshots": "captured_ts TEXT, captured_date TEXT, phase TEXT, game_date TEXT, "
                      "kickoff TEXT, home TEXT, away TEXT, au_point REAL, au_home_odds REAL, "
                      "au_away_odds REAL, au_h2h_home REAL, au_h2h_away REAL, au_total REAL, "
                      "au_over REAL, au_under REAL, sharp_point REAL, sharp_home_odds REAL, "
                      "sharp_away_odds REAL, sharp_total REAL, sharp_h2h_home REAL, sharp_h2h_away REAL",
    "book_lines": "captured_ts TEXT, phase TEXT, game_date TEXT, home TEXT, away TEXT, "
                  "book TEXT, title TEXT, region TEXT, sp_point REAL, sp_home REAL, sp_away REAL, "
                  "h2h_home REAL, h2h_away REAL, tot_point REAL, tot_over REAL, tot_under REAL",
    "predictions": "captured_date TEXT, captured_ts TEXT, phase TEXT, game_date TEXT, home TEXT, "
                   "away TEXT, model TEXT, pred_margin REAL, edge REAL, au_point REAL, "
                   "sharp_point REAL, win_prob REAL, out_diff REAL, backup_qb_diff REAL, "
                   "pred_total REAL, au_total REAL, sharp_total REAL",
    "model_coeffs": "captured_date TEXT, feature TEXT, coef REAL",
    "sent_alerts": "key TEXT PRIMARY KEY, sent_ts TEXT",
    "api_usage": "checked_at TEXT, used INTEGER, remaining INTEGER, last INTEGER",
    "kv": "key TEXT PRIMARY KEY, value TEXT",
    "bankroll_history": "date TEXT PRIMARY KEY, total REAL, unstaked REAL, pending REAL, realized REAL",
    "injury_log": "captured_ts TEXT, team TEXT, player TEXT, position TEXT, status TEXT, "
                  "detail TEXT, entered TEXT",
}


def init_db() -> None:
    conn = connect()
    try:
        for t, ddl in _SCHEMA.items():
            conn.execute(f'CREATE TABLE IF NOT EXISTS "{t}" ({ddl})')
        # migrations for tables created before a column existed
        cols = {r[1] for r in conn.execute('PRAGMA table_info("bets")')}
        if cols and "model" not in cols:
            conn.execute('ALTER TABLE bets ADD COLUMN model TEXT')
        pcols = {r[1] for r in conn.execute('PRAGMA table_info("predictions")')}
        for c in ("pred_total", "au_total", "sharp_total"):
            if pcols and c not in pcols:
                conn.execute(f'ALTER TABLE predictions ADD COLUMN {c} REAL')
        _ensure_matches_current(conn)
        conn.commit()
    finally:
        conn.close()


# --- matches ----------------------------------------------------------------
def _seed_matches(conn) -> None:
    from src.load_data import read_xlsx
    df = read_xlsx()
    for c in df.columns:
        if df[c].dtype == object:
            sample = df[c].dropna()
            if len(sample) and isinstance(sample.iloc[0], (datetime.time, datetime.timedelta)):
                df[c] = df[c].astype(str)
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce").dt.strftime("%Y-%m-%d")
    df.to_sql("matches", conn, if_exists="replace", index=False)


def _ensure_matches_current(conn) -> None:
    """(Re)seed `matches` when the workbook appears or changes (mtime tracked in kv)."""
    if not Path(RAW_XLSX).exists():
        return
    stamp = str(int(Path(RAW_XLSX).stat().st_mtime))
    if _has_rows(conn, "matches") and get_kv("xlsx_mtime", conn=conn) == stamp:
        return
    _seed_matches(conn)
    set_kv("xlsx_mtime", stamp, conn=conn)
    # auto-captured results now covered by the workbook are redundant
    if _has_table(conn, "auto_matches"):
        mx = conn.execute('SELECT MAX("Date") FROM matches').fetchone()[0]
        if mx:
            conn.execute('DELETE FROM auto_matches WHERE "Date" <= ?', (mx,))


def load_matches() -> pd.DataFrame | None:
    conn = connect()
    try:
        if not _has_rows(conn, "matches"):
            return None
        df = pd.read_sql('SELECT * FROM matches', conn)
        if _has_table(conn, "auto_matches") and _has_rows(conn, "auto_matches"):
            auto = pd.read_sql('SELECT * FROM auto_matches', conn)
            df = pd.concat([df, auto], ignore_index=True)
        return df
    finally:
        conn.close()


def capture_results(results: list) -> int:
    """Store finished games [{date, home, away, home_score, away_score}] not
    already known, pairing each with the AU opening/closing line we captured."""
    if not results:
        return 0
    conn = connect()
    n = 0
    try:
        known = set()
        for t in ("matches", "auto_matches"):
            if _has_table(conn, t):
                for d, h, a in conn.execute(f'SELECT "Date","Home Team","Away Team" FROM "{t}"'):
                    known.add((str(d)[:10], h, a))
        snaps = odds_snapshots_df(conn)
        rows = []
        for r in results:
            gd = str(pd.Timestamp(r["date"]).date())
            key = (gd, r["home"], r["away"])
            if key in known:
                continue
            g = snaps[(snaps["game_date"] == gd) & (snaps["home"] == r["home"])
                      & (snaps["away"] == r["away"])].sort_values("captured_ts") if len(snaps) else snaps
            first = g.iloc[0] if len(g) else None
            last = g.iloc[-1] if len(g) else None
            rows.append({
                "Date": gd, "Home Team": r["home"], "Away Team": r["away"],
                "Home Score": r["home_score"], "Away Score": r["away_score"],
                "Home Line Open": first["au_point"] if first is not None else None,
                "Home Line Close": last["au_point"] if last is not None else None,
                "Home Line Odds Open": first["au_home_odds"] if first is not None else None,
                "Away Line Odds Open": first["au_away_odds"] if first is not None else None,
                "Home Line Odds Close": last["au_home_odds"] if last is not None else None,
                "Away Line Odds Close": last["au_away_odds"] if last is not None else None,
                "Total Score Open": first["au_total"] if first is not None else None,
                "Total Score Close": last["au_total"] if last is not None else None,
                "Playoff Game?": "Y" if r.get("playoff") else None,
                "Neutral Venue?": "Y" if r.get("neutral") else None,
                "captured_ts": _now_iso(),
            })
            known.add(key)
            n += 1
        if rows:
            pd.DataFrame(rows).to_sql("auto_matches", conn, if_exists="append", index=False)
            conn.commit()
    finally:
        conn.close()
    return n


# --- outs -------------------------------------------------------------------
def read_outs() -> pd.DataFrame:
    conn = connect()
    try:
        df = pd.read_sql("SELECT rowid AS rowid, * FROM player_outs", conn)
    finally:
        conn.close()
    for c in OUT_COLUMNS:
        if c not in df.columns:
            df[c] = None
    return df


def add_out(entered, weeks, team, player, position, points, kind="out", source="manual", note="") -> None:
    conn = connect()
    try:
        conn.execute("INSERT INTO player_outs VALUES (?,?,?,?,?,?,?,?,?)",
                     (str(entered), float(weeks), team, player, position, float(points), kind, source, note))
        conn.commit()
    finally:
        conn.close()


def remove_out(rowid: int) -> None:
    conn = connect()
    try:
        conn.execute("DELETE FROM player_outs WHERE rowid=?", (int(rowid),))
        conn.commit()
    finally:
        conn.close()


def replace_source_outs(source: str, rows: list) -> int:
    """Atomically replace every out from `source` (e.g. 'espn') with `rows`.
    Manual rows are never touched."""
    conn = connect()
    try:
        conn.execute("DELETE FROM player_outs WHERE source=?", (source,))
        for r in rows:
            conn.execute("INSERT INTO player_outs VALUES (?,?,?,?,?,?,?,?,?)",
                         (str(r["entered"]), float(r.get("weeks", 1)), r["team"], r["player"],
                          r.get("position"), float(r.get("points", 0)), r.get("kind", "out"),
                          source, r.get("note", "")))
        conn.commit()
        return len(rows)
    finally:
        conn.close()


# --- bets -------------------------------------------------------------------
def read_bets() -> pd.DataFrame:
    conn = connect()
    try:
        return pd.read_sql("SELECT rowid AS rowid, * FROM bets", conn)
    finally:
        conn.close()


def add_bet(row: dict) -> None:
    conn = connect()
    try:
        cols = ",".join(BET_COLUMNS)
        conn.execute(f"INSERT INTO bets ({cols}) VALUES ({','.join('?' * len(BET_COLUMNS))})",
                     [row.get(c) for c in BET_COLUMNS])
        conn.commit()
    finally:
        conn.close()


def update_bet(rowid: int, **fields) -> None:
    if not fields:
        return
    conn = connect()
    try:
        sets = ",".join(f"{k}=?" for k in fields)
        conn.execute(f"UPDATE bets SET {sets} WHERE rowid=?", [*fields.values(), int(rowid)])
        conn.commit()
    finally:
        conn.close()


def remove_bet(rowid: int) -> None:
    conn = connect()
    try:
        conn.execute("DELETE FROM bets WHERE rowid=?", (int(rowid),))
        conn.commit()
    finally:
        conn.close()


# --- odds snapshots -----------------------------------------------------------
def snapshot_odds(events: list, phase: str = "page") -> int:
    """One row per game per phase per day (re-runs are no-ops)."""
    conn = connect()
    n = 0
    try:
        today = _today()
        for e in events:
            gd = str(pd.Timestamp(e["date"]).date())
            dup = conn.execute("SELECT 1 FROM odds_snapshots WHERE captured_date=? AND phase=? "
                               "AND game_date=? AND home=? AND away=?",
                               (today, phase, gd, e["home"], e["away"])).fetchone()
            if dup:
                continue
            conn.execute(
                "INSERT INTO odds_snapshots VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                (_now_iso(), today, phase, gd, str(e.get("kickoff")), e["home"], e["away"],
                 _f(e.get("au_point")), _f(e.get("au_home_odds")), _f(e.get("au_away_odds")),
                 _f(e.get("au_h2h_home")), _f(e.get("au_h2h_away")), _f(e.get("au_total")),
                 _f(e.get("au_over")), _f(e.get("au_under")), _f(e.get("sharp_point")),
                 _f(e.get("sharp_home_odds")), _f(e.get("sharp_away_odds")), _f(e.get("sharp_total")),
                 _f(e.get("sharp_h2h_home")), _f(e.get("sharp_h2h_away"))))
            n += 1
        conn.commit()
    finally:
        conn.close()
    return n


def snapshot_game(event: dict, phase: str) -> int:
    """One snapshot per game per phase EVER (used for the kickoff close)."""
    conn = connect()
    try:
        gd = str(pd.Timestamp(event["date"]).date())
        if conn.execute("SELECT 1 FROM odds_snapshots WHERE phase=? AND game_date=? AND home=? AND away=?",
                        (phase, gd, event["home"], event["away"])).fetchone():
            return 0
    finally:
        conn.close()
    return snapshot_odds([event], phase=phase)


def snapshot_book_lines(events: list, phase: str = "page") -> int:
    conn = connect()
    n = 0
    try:
        ts = _now_iso()
        for e in events:
            gd = str(pd.Timestamp(e["date"]).date())
            for b in e.get("books", []):
                conn.execute("INSERT INTO book_lines VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                             (ts, phase, gd, e["home"], e["away"], b.get("key"), b.get("title"),
                              b.get("region"), _f(b.get("sp_point")), _f(b.get("sp_home")),
                              _f(b.get("sp_away")), _f(b.get("h2h_home")), _f(b.get("h2h_away")),
                              _f(b.get("tot_point")), _f(b.get("tot_over")), _f(b.get("tot_under"))))
                n += 1
        conn.commit()
    finally:
        conn.close()
    return n


def snapshot_predictions(up: pd.DataFrame, phase: str = "page") -> int:
    """Capture the model's view once per game per day (for forward CLV/scorecards)."""
    if up is None or not len(up):
        return 0
    conn = connect()
    n = 0
    try:
        today = _today()
        for _, r in up.iterrows():
            gd = str(pd.Timestamp(r["Date"]).date())
            if conn.execute("SELECT 1 FROM predictions WHERE captured_date=? AND game_date=? AND home=? "
                            "AND away=? AND model='blend'", (today, gd, r["Home Team"], r["Away Team"])).fetchone():
                continue
            conn.execute("INSERT INTO predictions VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                         (today, _now_iso(), phase, gd, r["Home Team"], r["Away Team"], "blend",
                          _f(r.get("pred_margin")), _f(r.get("edge")), _f(r.get("au_point")),
                          _f(r.get("sharp_point")), _f(r.get("home_win_prob")),
                          _f(r.get("player_out_diff")), _f(r.get("backup_qb_diff")),
                          _f(r.get("pred_total")), _f(r.get("au_total")), _f(r.get("sharp_total"))))
            n += 1
        conn.commit()
    finally:
        conn.close()
    return n


def snapshot_model_coeffs(features, coefs) -> None:
    conn = connect()
    try:
        today = _today()
        if conn.execute("SELECT 1 FROM model_coeffs WHERE captured_date=?", (today,)).fetchone():
            return
        for f, c in zip(features, coefs):
            conn.execute("INSERT INTO model_coeffs VALUES (?,?,?)", (today, f, float(c)))
        conn.commit()
    finally:
        conn.close()


def _df(sql, conn=None) -> pd.DataFrame:
    own = conn is None
    conn = conn or connect()
    try:
        return pd.read_sql(sql, conn)
    except Exception:  # noqa: BLE001
        return pd.DataFrame()
    finally:
        if own:
            conn.close()


def odds_snapshots_df(conn=None) -> pd.DataFrame:
    return _df("SELECT * FROM odds_snapshots", conn)


def book_lines_df() -> pd.DataFrame:
    return _df("SELECT * FROM book_lines")


def predictions_df() -> pd.DataFrame:
    return _df("SELECT * FROM predictions")


def model_coeffs_df() -> pd.DataFrame:
    return _df("SELECT * FROM model_coeffs")


# --- alerts / usage / kv -----------------------------------------------------
def was_alerted(key: str) -> bool:
    conn = connect()
    try:
        return conn.execute("SELECT 1 FROM sent_alerts WHERE key=?", (key,)).fetchone() is not None
    finally:
        conn.close()


def mark_alerted(key: str) -> None:
    conn = connect()
    try:
        conn.execute("INSERT OR IGNORE INTO sent_alerts VALUES (?,?)", (key, _now_iso()))
        conn.commit()
    finally:
        conn.close()


def record_api_usage(used=None, remaining=None, last=None) -> None:
    conn = connect()
    try:
        conn.execute("INSERT INTO api_usage VALUES (?,?,?,?)",
                     (_now_iso(), _i(used), _i(remaining), _i(last)))
        conn.commit()
    finally:
        conn.close()


def api_usage() -> dict:
    conn = connect()
    try:
        r = conn.execute("SELECT checked_at, used, remaining, last FROM api_usage "
                         "ORDER BY checked_at DESC LIMIT 1").fetchone()
    finally:
        conn.close()
    if not r:
        return {}
    return {"checked_at": r[0], "used": r[1], "remaining": r[2], "last": r[3]}


def get_kv(key: str, default=None, conn=None):
    own = conn is None
    conn = conn or connect()
    try:
        r = conn.execute("SELECT value FROM kv WHERE key=?", (key,)).fetchone()
        return r[0] if r else default
    except Exception:  # noqa: BLE001
        return default
    finally:
        if own:
            conn.close()


def set_kv(key: str, value, conn=None) -> None:
    own = conn is None
    conn = conn or connect()
    try:
        conn.execute("INSERT OR REPLACE INTO kv VALUES (?,?)", (key, str(value)))
        if own:
            conn.commit()
    finally:
        if own:
            conn.close()


def get_json(key: str, default=None):
    v = get_kv(key)
    try:
        return json.loads(v) if v else default
    except Exception:  # noqa: BLE001
        return default


def set_json(key: str, obj) -> None:
    set_kv(key, json.dumps(obj))


def log_injuries(rows: list) -> None:
    if not rows:
        return
    conn = connect()
    try:
        ts = _now_iso()
        for r in rows:
            conn.execute("INSERT INTO injury_log VALUES (?,?,?,?,?,?,?)",
                         (ts, r.get("team"), r.get("player"), r.get("position"), r.get("status"),
                          r.get("detail"), r.get("entered")))
        conn.commit()
    finally:
        conn.close()


def _f(x):
    try:
        if x is None or pd.isna(x):
            return None
        return float(x)
    except Exception:  # noqa: BLE001
        return None


def _i(x):
    try:
        return int(x) if x is not None else None
    except Exception:  # noqa: BLE001
        return None
