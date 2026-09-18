"""ESPN injury feed -> player outs (with the starting-QB flag).

ESPN's public JSON lists every team's injuries with a game status: Out /
Doubtful / Questionable / Injured Reserve / Active. We convert Out, Doubtful
and IR into outs, sized by position prior (config.POSITION_POINTS). A QB is
only meaningful if he is the team's PRIMARY starter (mode of this season's
nflverse starting QBs) — then the row is tagged position QB so the model's
learned backup-QB coefficient prices it. Other QBs are ignored.

Questionable players are NOT outs (they play ~70% of the time) — they're logged
and shown as a watch-list. Rows are written with source="espn" and replaced
wholesale on every sync; manual rows are untouched.

IMPORTANT: use a plain User-Agent. ESPN 403s a spoofed browser UA from here.

    python -m src.injuries          # sync
    python -m src.injuries --dry    # print only
"""
import re
import sys

import pandas as pd
import requests

from config import (ESPN_INJURIES_URL, LOCAL_TZ, NON_QB_STARTER_PROB, OUT_STATUSES,
                    POSITION_POINTS)
from src import db
from src.load_data import load_nflverse
from src.teams import canon_team

# requests' default UA ("python-requests/x") is accepted by ESPN; custom or browser UAs get 403.
UA = {"Accept": "application/json"}


def _norm(s) -> str:
    return re.sub(r"[^a-z]", "", str(s or "").lower())


def primary_qbs() -> dict:
    """team -> primary starting QB (mode of this season's completed games; if
    < 2 games this season, last season's mode)."""
    nv = load_nflverse()
    out = {}
    if nv is None or not len(nv):
        return out
    nv = nv.dropna(subset=["home_score"])                     # completed games only
    nv["season"] = nv["gameday"].dt.year.where(nv["gameday"].dt.month >= 6, nv["gameday"].dt.year - 1)
    long = pd.concat([
        nv[["season", "gameday", "home", "home_qb_name"]].rename(columns={"home": "team", "home_qb_name": "qb"}),
        nv[["season", "gameday", "away", "away_qb_name"]].rename(columns={"away": "team", "away_qb_name": "qb"}),
    ]).dropna(subset=["qb"])
    for team, g in long.groupby("team"):
        seasons = sorted(g["season"].unique())
        if not seasons:
            continue
        cur = g[g["season"] == seasons[-1]]
        if len(cur) >= 2 or len(seasons) == 1:
            out[team] = cur["qb"].mode().iloc[0]
        else:
            prev = g[g["season"] == seasons[-2]]
            out[team] = prev["qb"].mode().iloc[0] if len(prev) else cur["qb"].mode().iloc[0]
    return out


def fetch_espn_injuries() -> list:
    r = requests.get(ESPN_INJURIES_URL, headers=UA, timeout=30)
    r.raise_for_status()
    data = r.json()
    rows = []
    for t in data.get("injuries", []):
        team = canon_team(t.get("displayName"))
        for i in t.get("injuries", []):
            ath = i.get("athlete", {})
            rows.append({
                "team": team,
                "player": ath.get("displayName"),
                "position": (ath.get("position", {}) or {}).get("abbreviation", ""),
                "status": (i.get("status") or "").strip(),
                "detail": (i.get("longComment") or i.get("shortComment") or "")[:300],
                "date": i.get("date"),
                "type": ((i.get("type") or {}).get("description") or ""),
            })
    return rows


def _entered(date_str) -> str:
    today = pd.Timestamp.now(tz=LOCAL_TZ).normalize().tz_localize(None)
    try:
        d = pd.to_datetime(date_str, utc=True).tz_convert(LOCAL_TZ).tz_localize(None).normalize()
        return str(min(d, today).date())
    except Exception:  # noqa: BLE001
        return str(today.date())


def build_outs(rows: list, qbs: dict = None) -> tuple[list, list]:
    """-> (outs to write, watch-list rows)."""
    qbs = qbs if qbs is not None else primary_qbs()
    outs, watch = [], []
    for r in rows:
        st = r["status"].lower()
        pos = (r["position"] or "").upper()
        if st == "questionable":
            watch.append(r)
            continue
        if st not in OUT_STATUSES:
            continue
        long_term = st in ("injured reserve", "ir", "pup", "nfi")
        weeks = 6.0 if long_term else 1.0
        scale = 0.75 if st == "doubtful" else 1.0
        if pos == "QB":
            primary = qbs.get(r["team"])
            if primary and _norm(primary) == _norm(r["player"]):
                outs.append({"entered": _entered(r["date"]), "weeks": weeks, "team": r["team"],
                             "player": r["player"], "position": "QB",
                             "points": POSITION_POINTS["QB"] * scale, "kind": "out",
                             "note": f"{r['status']} (starter) — {r['detail'][:120]}"})
            else:
                watch.append({**r, "status": f"{r['status']} (backup QB, ignored)"})
            continue
        pts = POSITION_POINTS.get(pos, 0.5) * NON_QB_STARTER_PROB * scale
        outs.append({"entered": _entered(r["date"]), "weeks": weeks, "team": r["team"],
                     "player": r["player"], "position": pos or "?", "points": round(pts, 2),
                     "kind": "out", "note": f"{r['status']} — {r['detail'][:120]}"})
    return outs, watch


def sync(dry: bool = False) -> dict:
    rows = fetch_espn_injuries()
    qbs = primary_qbs()
    outs, watch = build_outs(rows, qbs)
    status = {"ts": pd.Timestamp.now(tz=LOCAL_TZ).strftime("%Y-%m-%d %H:%M"),
              "teams": len({r["team"] for r in rows}), "listed": len(rows),
              "outs": len(outs), "questionable": len(watch), "ok": len(rows) > 0}
    if dry:
        for o in outs:
            print(f"OUT  {o['team']:<24} {o['position']:<4} {o['player']:<26} {o['points']:>4.2f}  {o['note'][:60]}")
        print(f"\n{len(outs)} outs, {len(watch)} questionable/other")
        return status
    db.replace_source_outs("espn", outs)
    db.log_injuries([{"team": r["team"], "player": r["player"], "position": r["position"],
                      "status": r["status"], "detail": r["detail"][:120], "entered": _entered(r["date"])}
                     for r in rows if r["status"].lower() != "active"])
    db.set_json("injury_status", status)
    db.set_json("injury_watch", [{"team": w["team"], "player": w["player"], "position": w["position"],
                                  "status": w["status"], "detail": (w.get("detail") or "")[:100]} for w in watch])
    db.set_json("primary_qbs", qbs)
    print(f"injuries: {len(outs)} outs written (espn), {len(watch)} on watch-list")
    return status


if __name__ == "__main__":
    db.init_db()
    sync(dry="--dry" in sys.argv)
