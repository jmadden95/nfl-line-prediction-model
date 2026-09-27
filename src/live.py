"""Live NFL fixtures + odds from The Odds API -> model predictions.

One call fetches every book in ODDS_REGIONS (au + eu). From each event we keep:
  au_*      the AU_BOOK's (Sportsbet) line/prices — the line we can BET
  sharp_*   the sharp reference (Pinnacle) — what the line "should" be
  books     every book's prices (region-tagged) for lag analysis + shopping

Dates: `date` is the US-Eastern game date (matches the workbook); `kickoff` is
the kickoff in LOCAL_TZ (for display + the kickoff-timed close capture).

    python -m src.live --mock
    python -m src.live
"""
import json
import os
import sys
import time

import numpy as np
import pandas as pd
import requests
from sklearn.linear_model import LinearRegression

from config import (AU_BOOK, AU_BOOK_KEYS, DATA_DIR, LOCAL_TZ, MARKET_BLEND, ODDS_API_KEY,
                    ODDS_CACHE_TTL, ODDS_MARKETS, ODDS_REGIONS, ODDS_SPORT, SHARP_BOOKS,
                    TRAIN_FROM_SEASON)
from src.features import FEATURE_COLUMNS, LINE_FREE_COLUMNS, build_features
from src.load_data import add_margin, load_raw, season_of
from src.teams import canon_team
from src.win_prob import margin_to_win_prob

ODDS_API_URL = f"https://api.the-odds-api.com/v4/sports/{ODDS_SPORT}/odds"

ODDS_CACHE_PATH = DATA_DIR / "odds_cache.json"
US_TZ = "America/New_York"


def _med(xs):
    xs = [float(x) for x in xs if x is not None and x == x]
    return float(np.median(xs)) if xs else np.nan


def _parse_event(e: dict) -> dict:
    home, away = canon_team(e["home_team"]), canon_team(e["away_team"])
    books = []
    for bk in e.get("bookmakers", []):
        key = bk.get("key")
        rec = {"key": key, "title": bk.get("title") or key,
               "region": "au" if key in AU_BOOK_KEYS else "other",
               "sharp": key in SHARP_BOOKS}
        for mk in bk.get("markets", []):
            outs = {canon_team(o["name"]) if o["name"] not in ("Over", "Under") else o["name"]: o
                    for o in mk.get("outcomes", [])}
            k = mk.get("key")
            if k == "h2h" and home in outs and away in outs:
                rec["h2h_home"], rec["h2h_away"] = outs[home].get("price"), outs[away].get("price")
            elif k == "spreads" and home in outs and away in outs:
                rec["sp_point"] = outs[home].get("point")
                rec["sp_home"], rec["sp_away"] = outs[home].get("price"), outs[away].get("price")
            elif k == "totals" and "Over" in outs and "Under" in outs:
                rec["tot_point"] = outs["Over"].get("point")
                rec["tot_over"], rec["tot_under"] = outs["Over"].get("price"), outs["Under"].get("price")
        if len(rec) > 4:
            books.append(rec)

    au = next((b for b in books if b["key"] == AU_BOOK), None)
    au_all = [b for b in books if b["region"] == "au"]
        # Sharp reference: SHARP_BOOKS (Pinnacle) only. If it hasn't posted this
    # game yet there is NO sharp line (no fallback to softer books).
    sharps = [b for b in books if b["sharp"]]
    sharp_label = None

    def pick(rec, k):
        return rec.get(k, np.nan) if rec else np.nan

    def med_of(recs, k):
        return _med([b.get(k) for b in recs])

    ko_utc = pd.to_datetime(e["commence_time"], utc=True)
    ko_local = ko_utc.tz_convert(LOCAL_TZ).tz_localize(None)
    us_date = ko_utc.tz_convert(US_TZ).tz_localize(None).normalize()
    return {
        "id": e.get("id"), "date": us_date, "kickoff": ko_local, "kickoff_utc": ko_utc,
        "home": home, "away": away,
        "au_book": au["title"] if au else (au_all[0]["title"] + " (fallback)" if au_all else None),
        "au_point": pick(au, "sp_point") if au else med_of(au_all, "sp_point"),
        "au_home_odds": pick(au, "sp_home") if au else med_of(au_all, "sp_home"),
        "au_away_odds": pick(au, "sp_away") if au else med_of(au_all, "sp_away"),
        "au_h2h_home": pick(au, "h2h_home") if au else med_of(au_all, "h2h_home"),
        "au_h2h_away": pick(au, "h2h_away") if au else med_of(au_all, "h2h_away"),
        "au_total": pick(au, "tot_point") if au else med_of(au_all, "tot_point"),
        "au_over": pick(au, "tot_over") if au else med_of(au_all, "tot_over"),
        "au_under": pick(au, "tot_under") if au else med_of(au_all, "tot_under"),
        "sharp_book": (sharp_label or ", ".join(sorted({b["title"] for b in sharps}))) if sharps else None,
        "sharp_detail": {b["title"]: (b.get("sp_point"), b.get("tot_point")) for b in sharps},
        "sharp_point": med_of(sharps, "sp_point"),
        "sharp_home_odds": med_of(sharps, "sp_home"),
        "sharp_away_odds": med_of(sharps, "sp_away"),
        "sharp_total": med_of(sharps, "tot_point"),
        "sharp_h2h_home": med_of(sharps, "h2h_home"),
        "sharp_h2h_away": med_of(sharps, "h2h_away"),
        "au_consensus_point": med_of(au_all, "sp_point"),
        "n_au_books": len(au_all), "n_books": len(books),
        "books": books,
    }


def _record_usage(resp) -> None:
    try:
        from src import db
        db.record_api_usage(used=resp.headers.get("x-requests-used"),
                            remaining=resp.headers.get("x-requests-remaining"),
                            last=resp.headers.get("x-requests-last"))
    except Exception:  # noqa: BLE001
        pass


def cache_age_seconds():
    try:
        return time.time() - ODDS_CACHE_PATH.stat().st_mtime
    except Exception:  # noqa: BLE001
        return None


def fetch_odds(api_key: str = None, use_cache: bool = True) -> list:
    api_key = api_key or ODDS_API_KEY
    if not api_key:
        raise RuntimeError("No ODDS_API_KEY in .env (free key at https://the-odds-api.com)")
    age = cache_age_seconds()
    if use_cache and ODDS_CACHE_TTL > 0 and age is not None and age < ODDS_CACHE_TTL:
        try:
            return [_parse_event(e) for e in json.loads(ODDS_CACHE_PATH.read_text())]
        except Exception:  # noqa: BLE001
            pass
    params = {"apiKey": api_key, "regions": ODDS_REGIONS, "markets": ODDS_MARKETS,
              "oddsFormat": "decimal"}
    try:
        resp = requests.get(ODDS_API_URL, params=params, timeout=25)
        resp.raise_for_status()
        _record_usage(resp)
        raw = resp.json()
    except Exception:  # noqa: BLE001
        if ODDS_CACHE_PATH.exists():
            return [_parse_event(e) for e in json.loads(ODDS_CACHE_PATH.read_text())]
        raise
    try:
        ODDS_CACHE_PATH.write_text(json.dumps(raw))
    except Exception:  # noqa: BLE001
        pass
    return [_parse_event(e) for e in raw]


def cached_events() -> list:
    """Parsed events from the cache only (no API call); [] if none."""
    try:
        return [_parse_event(e) for e in json.loads(ODDS_CACHE_PATH.read_text())]
    except Exception:  # noqa: BLE001
        return []


def mock_events() -> list:
    ko = pd.Timestamp.now(tz="UTC") + pd.Timedelta(days=2)
    return [_parse_event({
        "id": "mock", "commence_time": ko.isoformat(), "home_team": "Kansas City Chiefs",
        "away_team": "Buffalo Bills",
        "bookmakers": [
            {"key": "sportsbet", "title": "Sportsbet", "markets": [
                {"key": "spreads", "outcomes": [{"name": "Kansas City Chiefs", "point": -2.5, "price": 1.9},
                                                {"name": "Buffalo Bills", "point": 2.5, "price": 1.9}]},
                {"key": "h2h", "outcomes": [{"name": "Kansas City Chiefs", "price": 1.65},
                                            {"name": "Buffalo Bills", "price": 2.3}]},
                {"key": "totals", "outcomes": [{"name": "Over", "point": 47.5, "price": 1.9},
                                               {"name": "Under", "point": 47.5, "price": 1.9}]}]},
            {"key": "pinnacle", "title": "Pinnacle", "markets": [
                {"key": "spreads", "outcomes": [{"name": "Kansas City Chiefs", "point": -1.5, "price": 1.95},
                                                {"name": "Buffalo Bills", "point": 1.5, "price": 1.95}]}]},
        ]})]


# --- prediction -------------------------------------------------------------
def _upcoming_frame(events: list) -> pd.DataFrame:
    try:
        from src.weather import lookup, venue_for
        wx = lookup()
    except Exception:  # noqa: BLE001
        wx, venue_for = {}, None
    rows = []
    for e in events:
        gd = str(pd.Timestamp(e["date"]).date())
        w = wx.get((gd, e["home"], e["away"]), {})
        v = venue_for(e["home"], e["away"], e["date"]) if venue_for else {"enclosed": None, "neutral": False, "venue": None}
        enclosed = v.get("enclosed")
        wind = 0.0 if enclosed else w.get("wind_mph")
        rows.append({"Date": e["date"], "Kickoff": e.get("kickoff"), "Home Team": e["home"],
                     "Away Team": e["away"], "raw_home": e["home"], "raw_away": e["away"],
                     "Home Score": np.nan, "Away Score": np.nan,
                     "Home Line Open": e["au_point"], "Home Line Close": np.nan,
                     "is_playoff": 0.0, "neutral": 1.0 if v.get("neutral") else 0.0, "is_upcoming": True,
                     # roof drives is_dome (so neutral venues use OUR venue table, not the home dome)
                     "roof": ("closed" if enclosed else ("outdoors" if enclosed is False else np.nan)),
                     "wind": wind if wind is not None else np.nan,
                     "venue": v.get("venue"), "enclosed": enclosed,
                     "gust_mph": w.get("gust_mph"), "wx_captured": w.get("captured_ts")})
    df = pd.DataFrame(rows)
    if len(df):
        df["season"] = season_of(df["Date"])
    return df


def predict_upcoming(events: list, outs: pd.DataFrame = None) -> pd.DataFrame:
    hist = load_raw()
    hist["is_upcoming"] = False
    up_frame = _upcoming_frame(events)
    combined = pd.concat([hist, up_frame], ignore_index=True)
    combined = combined.sort_values(["Date", "Home Team"]).reset_index(drop=True)
    feat = build_features(add_margin(combined), outs)

    train = feat[(~feat["is_upcoming"]) & (feat["season"] >= TRAIN_FROM_SEASON)].dropna(
        subset=["home_margin"]).copy()
    train[FEATURE_COLUMNS] = train[FEATURE_COLUMNS].fillna(0.0)
    up = feat[feat["is_upcoming"]].copy()
    up[FEATURE_COLUMNS] = up[FEATURE_COLUMNS].fillna(0.0)
    if not len(up):
        return up

    model = LinearRegression().fit(train[LINE_FREE_COLUMNS], train["home_margin"])
    raw = model.predict(up[LINE_FREE_COLUMNS]) + up["player_out_diff"].to_numpy()
    up["pred_raw"] = raw
    bm = up["bookie_margin"].to_numpy(dtype=float)
    up["pred_margin"] = np.where(np.isnan(bm), raw, (1 - MARKET_BLEND) * raw + MARKET_BLEND * bm)
    up["edge"] = up["pred_margin"] - up["bookie_margin"]
    up["home_win_prob"] = margin_to_win_prob(up["pred_margin"])
    up["qb_coef"] = float(dict(zip(LINE_FREE_COLUMNS, model.coef_)).get("backup_qb_diff", 0.0))

    # attach the live market fields by (home, away)
    ev = {(e["home"], e["away"]): e for e in events}
    def col(k):
        return [ev.get((h, a), {}).get(k, np.nan) for h, a in zip(up["Home Team"], up["Away Team"])]
    for k in ["au_book", "au_point", "au_home_odds", "au_away_odds", "au_h2h_home", "au_h2h_away",
              "au_total", "au_over", "au_under", "sharp_book", "sharp_point", "sharp_home_odds",
              "sharp_away_odds", "sharp_total", "sharp_h2h_home", "sharp_h2h_away",
              "au_consensus_point", "n_au_books", "n_books", "books", "kickoff_utc", "sharp_detail"]:
        up[k] = col(k)
    # AU vs sharp disagreement, as home margin: + means sharps like home MORE than
    # the AU book does -> the AU home line is the value side (and vice versa).
    up["au_vs_sharp"] = (-pd.to_numeric(up["sharp_point"], errors="coerce")
                         + pd.to_numeric(up["au_point"], errors="coerce"))

    # keep only games that haven't kicked off
    now = pd.Timestamp.now(tz=LOCAL_TZ).tz_localize(None)
    ko = pd.to_datetime(up["Kickoff"], errors="coerce")
    up = up[ko.isna() | (ko > now)].copy()

    try:
        from src import db
        db.snapshot_model_coeffs(LINE_FREE_COLUMNS + ["intercept"], list(model.coef_) + [model.intercept_])
    except Exception:  # noqa: BLE001
        pass
    return up.sort_values("Kickoff").reset_index(drop=True)


def line_phrase(home, away, margin) -> str:
    if margin is None or margin != margin:
        return "n/a"
    return f"{home.split()[-1]} -{margin:.1f}" if margin >= 0 else f"{away.split()[-1]} -{abs(margin):.1f}"


def main() -> None:
    events = mock_events() if "--mock" in sys.argv else fetch_odds(use_cache="--fresh" not in sys.argv)
    up = predict_upcoming(events)
    print(f"{len(up)} upcoming games\n")
    for _, r in up.iterrows():
        print(f"{r['Kickoff']:%a %d %b %H:%M}  {r['Home Team']} v {r['Away Team']}")
        print(f"   AU {r.get('au_book')}: {line_phrase(r['Home Team'], r['Away Team'], r['bookie_margin'])}"
              f"   sharp {r.get('sharp_book')}: {line_phrase(r['Home Team'], r['Away Team'], -r['sharp_point'] if r['sharp_point']==r['sharp_point'] else np.nan)}"
              f"   AU-vs-sharp {r['au_vs_sharp']:+.1f}")
        print(f"   model {line_phrase(r['Home Team'], r['Away Team'], r['pred_margin'])}  edge {r['edge']:+.1f}"
              f"  P(home) {r['home_win_prob']:.0%}  outs {r['player_out_diff']:+.1f}  bkQB {r['backup_qb_diff']:+.0f}")


if __name__ == "__main__":
    main()
