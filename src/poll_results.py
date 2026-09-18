"""Auto-capture finished games from ESPN's scoreboard (free, no Odds API credits).

Stores each completed game (US-Eastern date, canonical names) in auto_matches,
paired with the AU opening/closing line we captured — so the model keeps
training itself and bets auto-settle without a workbook re-download.

    python -m src.poll_results            # last 4 days
    python -m src.poll_results --days 10
"""
import argparse

import pandas as pd
import requests

from config import ESPN_SCOREBOARD_URL
from src import db
from src.teams import canon_team

# requests' default UA ("python-requests/x") is accepted by ESPN; custom or browser UAs get 403.
UA = {"Accept": "application/json"}


def fetch_scores(days: int = 4) -> list:
    out, seen = [], set()
    today = pd.Timestamp.now(tz="America/New_York").normalize()
    for d in range(days + 1):
        day = (today - pd.Timedelta(days=d)).strftime("%Y%m%d")
        try:
            data = requests.get(ESPN_SCOREBOARD_URL, params={"dates": day}, headers=UA, timeout=30).json()
        except Exception:  # noqa: BLE001
            continue
        for e in data.get("events", []):
            if e.get("id") in seen:
                continue
            seen.add(e.get("id"))
            comp = (e.get("competitions") or [{}])[0]
            if not (comp.get("status", {}).get("type", {}).get("completed")):
                continue
            teams = {c.get("homeAway"): c for c in comp.get("competitors", [])}
            h, a = teams.get("home"), teams.get("away")
            if not h or not a:
                continue
            try:
                hs, as_ = float(h.get("score")), float(a.get("score"))
            except (TypeError, ValueError):
                continue
            gd = pd.to_datetime(e["date"], utc=True).tz_convert("America/New_York").tz_localize(None).normalize()
            out.append({"date": gd, "home": canon_team(h["team"]["displayName"]),
                        "away": canon_team(a["team"]["displayName"]), "home_score": hs, "away_score": as_,
                        "neutral": bool(comp.get("neutralSite")),
                        "playoff": (e.get("season", {}) or {}).get("type") == 3})
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--days", type=int, default=4)
    args = ap.parse_args()
    db.init_db()
    res = fetch_scores(args.days)
    n = db.capture_results(res)
    print(f"poll_results: {len(res)} completed games fetched, {n} new captured")
