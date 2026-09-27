"""Game-time wind for the totals model — enclosed stadiums get ZERO wind.

History: nflverse's per-game `wind` (mph, observed) and `roof`. Wind lowers NFL
totals ~0.26 pts per mph (walk-forward fit), and the OPENING total under-prices
it: games at >= 15 mph went under the open 60% of the time (2012-2025), because
the opener is posted before the wind forecast exists. That is the NFL twin of
the NRL rain edge — it only pays if you bet before the market catches up.

Live: open-meteo hourly forecast (free, no key) at the game's VENUE for the
kickoff-to-+3h window. Enclosure rules:
  * fixed domes and RETRACTABLE roofs -> enclosed, wind 0 (retractables close
    when it's windy/wet, so an open-air forecast would mislead)
  * neutral / international games use OUR venue table (nflverse's future-game
    roof is wrong there: it lists the MCG, Stade de France and Munich as domes)
  * unknown neutral venue -> no forecast (wind treated as missing, not 0)

    python -m src.weather          # capture forecasts for upcoming games + print
"""
from __future__ import annotations

import pandas as pd
import requests

from config import LOCAL_TZ
from src import db
from src.teams import TEAM_DOME, TEAM_HOME, canon_team

FORECAST_URL = "https://api.open-meteo.com/v1/forecast"
THROTTLE_HOURS = 3.0

# Neutral / international venues: name fragment -> (lat, lon, enclosed)
VENUES = {
    "tottenham": (51.604, -0.066, False),
    "wembley": (51.556, -0.280, False),
    "bayern": (48.219, 11.625, False), "allianz arena": (48.219, 11.625, False),
    "olympiastadion": (52.515, 13.239, False),
    "frankfurt": (50.069, 8.646, False), "deutsche bank park": (50.069, 8.646, False),
    "stade de france": (48.924, 2.360, False),
    "melbourne cricket ground": (-37.820, 144.983, False), "mcg": (-37.820, 144.983, False),
    "banorte": (19.303, -99.150, False), "azteca": (19.303, -99.150, False),
    "maracana": (-22.912, -43.230, False),
    "corinthians": (-23.545, -46.474, False), "neo quimica": (-23.545, -46.474, False),
    "bernabeu": (40.453, -3.688, True),          # retractable roof
    "croke park": (53.361, -6.251, False),
}


def _nflverse_row(home: str, away: str, us_date) -> dict | None:
    try:
        from src.load_data import load_nflverse
        nv = load_nflverse()
    except Exception:  # noqa: BLE001
        return None
    if nv is None:
        return None
    d = pd.Timestamp(us_date).normalize()
    hit = nv[(nv["gameday"].dt.normalize() == d) & (nv["home"] == home) & (nv["away"] == away)]
    return hit.iloc[0].to_dict() if len(hit) else None


def venue_for(home: str, away: str, us_date) -> dict:
    """-> {venue, lat, lon, enclosed, neutral, source}."""
    home = canon_team(home)
    row = _nflverse_row(home, away, us_date)
    stadium = str((row or {}).get("stadium") or "")
    neutral = str((row or {}).get("location") or "").lower() == "neutral"
    if neutral:
        key = next((k for k in VENUES if k in stadium.lower()), None)
        if key is None:
            return {"venue": stadium or "neutral (unknown)", "lat": None, "lon": None,
                    "enclosed": None, "neutral": True, "source": "unknown neutral venue"}
        lat, lon, enc = VENUES[key]
        return {"venue": stadium, "lat": lat, "lon": lon, "enclosed": enc, "neutral": True, "source": "venue table"}
    lat, lon = TEAM_HOME.get(home, (None, None))
    return {"venue": stadium or f"{home} home", "lat": lat, "lon": lon,
            "enclosed": bool(TEAM_DOME.get(home, False)), "neutral": False, "source": "home stadium"}


def forecast(lat, lon, kickoff_utc) -> dict | None:
    """Mean wind/gust (mph), temp (F), precip (mm) over kickoff..kickoff+3h."""
    ko = pd.Timestamp(kickoff_utc)
    ko = ko.tz_localize("UTC") if ko.tzinfo is None else ko.tz_convert("UTC")
    try:
        r = requests.get(FORECAST_URL, params={
            "latitude": lat, "longitude": lon, "timezone": "UTC",
            "hourly": "wind_speed_10m,wind_gusts_10m,temperature_2m,precipitation",
            "wind_speed_unit": "mph", "temperature_unit": "fahrenheit",
            "start_date": str(ko.date()), "end_date": str((ko + pd.Timedelta(hours=4)).date())}, timeout=30)
        h = r.json().get("hourly", {})
        t = pd.to_datetime(h.get("time", []), utc=True)
        f = pd.DataFrame({"t": t, "wind": h.get("wind_speed_10m"), "gust": h.get("wind_gusts_10m"),
                          "temp": h.get("temperature_2m"), "precip": h.get("precipitation")})
        w = f[(f["t"] >= ko.floor("h")) & (f["t"] <= ko + pd.Timedelta(hours=3))]
        if not len(w):
            return None
        return {"wind_mph": float(w["wind"].mean()), "gust_mph": float(w["gust"].mean()),
                "temp_f": float(w["temp"].mean()), "precip_mm": float(w["precip"].sum())}
    except Exception:  # noqa: BLE001
        return None


def _ensure(conn):
    conn.execute("CREATE TABLE IF NOT EXISTS weather_forecast(captured_ts TEXT, game_date TEXT, home TEXT, "
                 "away TEXT, venue TEXT, enclosed INTEGER, neutral INTEGER, wind_mph REAL, gust_mph REAL, "
                 "temp_f REAL, precip_mm REAL, source TEXT)")


def capture_forecast(events: list, force: bool = False) -> int:
    """Store a forecast per upcoming game (throttled to one per THROTTLE_HOURS)."""
    now = pd.Timestamp.now(tz="UTC")
    conn = db.connect()
    n = 0
    try:
        _ensure(conn)
        last = {(r[0], r[1], r[2]): r[3] for r in conn.execute(
            "SELECT game_date, home, away, MAX(captured_ts) FROM weather_forecast GROUP BY 1,2,3")}
        for e in events or []:
            ko = e.get("kickoff_utc")
            if ko is None or pd.Timestamp(ko) <= now or pd.Timestamp(ko) > now + pd.Timedelta(days=15):
                continue
            gd = str(pd.Timestamp(e["date"]).date())
            prev = last.get((gd, e["home"], e["away"]))
            if prev and not force and (pd.Timestamp.now(tz=LOCAL_TZ).tz_localize(None) - pd.Timestamp(prev)).total_seconds() < THROTTLE_HOURS * 3600:
                continue
            v = venue_for(e["home"], e["away"], e["date"])
            if v["enclosed"]:
                fc = {"wind_mph": 0.0, "gust_mph": 0.0, "temp_f": None, "precip_mm": 0.0}
            elif v["lat"] is None:
                fc = {"wind_mph": None, "gust_mph": None, "temp_f": None, "precip_mm": None}
            else:
                fc = forecast(v["lat"], v["lon"], ko) or {"wind_mph": None, "gust_mph": None, "temp_f": None, "precip_mm": None}
            conn.execute("INSERT INTO weather_forecast VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
                         (pd.Timestamp.now(tz=LOCAL_TZ).strftime("%Y-%m-%dT%H:%M:%S"), gd, e["home"], e["away"],
                          v["venue"], None if v["enclosed"] is None else int(v["enclosed"]), int(v["neutral"]),
                          fc["wind_mph"], fc["gust_mph"], fc["temp_f"], fc["precip_mm"], v["source"]))
            n += 1
        conn.commit()
    finally:
        conn.close()
    return n


def lookup() -> dict:
    """(game_date, home, away) -> latest forecast row dict."""
    conn = db.connect()
    try:
        _ensure(conn)
        df = pd.read_sql("SELECT * FROM weather_forecast", conn)
    finally:
        conn.close()
    if not len(df):
        return {}
    df = df.sort_values("captured_ts").groupby(["game_date", "home", "away"]).tail(1)
    return {(r["game_date"], r["home"], r["away"]): r.to_dict() for _, r in df.iterrows()}


if __name__ == "__main__":
    from src.live import cached_events
    db.init_db()
    print(f"captured {capture_forecast(cached_events(), force=True)} forecasts")
    for k, v in sorted(lookup().items()):
        tag = "ENCLOSED" if v["enclosed"] == 1 else ("unknown venue" if v["enclosed"] is None else f"wind {v['wind_mph']:.0f} mph (gust {v['gust_mph']:.0f})" if v["wind_mph"] == v["wind_mph"] and v["wind_mph"] is not None else "no forecast yet")
        print(f"  {k[0]} {k[2]:<24} @ {k[1]:<24} {v['venue'][:28]:<28} {tag}")
