"""Load + tidy the NFL results/odds workbook, enrich from nflverse, add the target.

The aussportsbetting NFL workbook has the SAME columns as the NRL one (Date,
Home Team, Away Team, Home/Away Score, Home Line Open/Close, odds, totals...)
but its header sits on the FIRST row (header=0), and ~70 rows carry the date as
text — we parse leniently.

nflverse games.csv adds, per game: week, rest days, roof, division flag, and
the STARTING QB names — the input that lets the model learn a backup-QB effect.

    python -m src.load_data     # sanity check
"""
from __future__ import annotations

import pandas as pd
import numpy as np

from config import RAW_XLSX, NFLVERSE_GAMES_CSV
from src.teams import TEAM_NAME_FIXES, canon_team, ABBR_TO_NAME


def season_of(dates: pd.Series) -> pd.Series:
    """NFL season = the calendar year it STARTS (Jan/Feb playoffs belong to the
    previous year's season)."""
    d = pd.to_datetime(dates, errors="coerce")
    return d.dt.year.where(d.dt.month >= 6, d.dt.year - 1)


def read_xlsx(path=None) -> pd.DataFrame:
    df = pd.read_excel(path or RAW_XLSX, sheet_name="Data", header=0)
    df.columns = [str(c).strip() for c in df.columns]
    return df


def tidy(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns]
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df = df.dropna(subset=["Date", "Home Team", "Away Team"])
    df["raw_home"] = df["Home Team"].astype(str).str.strip()
    df["raw_away"] = df["Away Team"].astype(str).str.strip()
    df["Home Team"] = df["raw_home"].map(lambda s: canon_team(TEAM_NAME_FIXES.get(s, s)))
    df["Away Team"] = df["raw_away"].map(lambda s: canon_team(TEAM_NAME_FIXES.get(s, s)))
    for c in ["Home Score", "Away Score", "Home Line Open", "Home Line Close",
              "Home Line Odds Open", "Away Line Odds Open", "Home Line Odds Close",
              "Away Line Odds Close", "Total Score Open", "Total Score Close",
              "Home Odds Open", "Away Odds Open", "Home Odds Close", "Away Odds Close"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    df["is_playoff"] = (df.get("Playoff Game?", pd.Series(index=df.index)).astype(str).str.upper() == "Y").astype(float)
    df["neutral"] = (df.get("Neutral Venue?", pd.Series(index=df.index)).astype(str).str.upper() == "Y").astype(float)
    df["season"] = season_of(df["Date"])
    # oldest first — every rolling feature assumes chronological order
    df = df.sort_values(["Date", "Home Team"]).reset_index(drop=True)
    return df


def load_raw() -> pd.DataFrame:
    """Matches from the SQLite store (seeded from the workbook + auto-captured
    results), falling back to the workbook itself."""
    df = None
    try:
        from src import db
        df = db.load_matches()
    except Exception:  # noqa: BLE001
        df = None
    if df is None:
        df = read_xlsx()
    return enrich_nflverse(tidy(df))


# --- nflverse enrichment ----------------------------------------------------
_NV_COLS = ["gameday", "home_team", "away_team", "week", "game_type", "home_rest",
            "away_rest", "roof", "surface", "div_game", "home_qb_name", "away_qb_name",
            "spread_line", "total_line", "home_score", "away_score", "temp", "wind",
            "stadium", "gametime"]


def load_nflverse() -> pd.DataFrame | None:
    """The cached nflverse games.csv (refreshed by src.nflverse_sync)."""
    try:
        nv = pd.read_csv(NFLVERSE_GAMES_CSV, usecols=lambda c: c in _NV_COLS, low_memory=False)
    except Exception:  # noqa: BLE001
        return None
    nv["gameday"] = pd.to_datetime(nv["gameday"], errors="coerce")
    nv["home"] = nv["home_team"].map(lambda a: ABBR_TO_NAME.get(str(a).upper(), a))
    nv["away"] = nv["away_team"].map(lambda a: ABBR_TO_NAME.get(str(a).upper(), a))
    return nv


def enrich_nflverse(df: pd.DataFrame) -> pd.DataFrame:
    """Left-join nflverse per-game context onto the workbook rows.

    Key: (date, home, away). US game dates: the workbook is dated by the US
    calendar day too (Monday Night = Monday), so dates line up directly. The
    Super Bowl / neutral games sometimes swap home/away between sources, so
    we also try the reversed pairing."""
    nv = load_nflverse()
    df = df.copy()
    add = ["week", "home_rest", "away_rest", "roof", "div_game",
           "home_qb_name", "away_qb_name", "nv_spread", "nv_total", "wind", "temp"]
    for c in add:
        df[c] = np.nan
    if nv is None or not len(nv):
        return df
    nv = nv.rename(columns={"spread_line": "nv_spread", "total_line": "nv_total"})
    keyed = nv.set_index([nv["gameday"].dt.normalize(), "home", "away"])
    keyed = keyed[~keyed.index.duplicated(keep="first")]
    rev = nv.copy()
    rev = rev.rename(columns={"home_rest": "away_rest", "away_rest": "home_rest",
                              "home_qb_name": "away_qb_name", "away_qb_name": "home_qb_name"})
    rev["nv_spread"] = -rev["nv_spread"]
    keyed_rev = rev.set_index([rev["gameday"].dt.normalize(), "away", "home"])
    keyed_rev = keyed_rev[~keyed_rev.index.duplicated(keep="first")]

    idx = pd.MultiIndex.from_arrays([df["Date"].dt.normalize(), df["Home Team"], df["Away Team"]])
    hit = keyed.reindex(idx)
    miss = hit["week"].isna().to_numpy()
    if miss.any():
        hit_rev = keyed_rev.reindex(idx)
        for c in add:
            if c in hit.columns:
                hit.loc[miss, c] = hit_rev.loc[miss, c].to_numpy() if c in hit_rev.columns else np.nan
    for c in add:
        if c in hit.columns:
            df[c] = hit[c].to_numpy()
    return df


def add_margin(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["home_margin"] = df["Home Score"] - df["Away Score"]
    df["total_points"] = df["Home Score"] + df["Away Score"]
    return df


if __name__ == "__main__":
    data = add_margin(load_raw())
    print(f"Loaded {len(data)} games, {data['Date'].min():%Y-%m-%d} -> {data['Date'].max():%Y-%m-%d}")
    print("nflverse match rate:", round(data["week"].notna().mean(), 3),
          "| QB names:", round(data["home_qb_name"].notna().mean(), 3))
    print(data.groupby("season").size().to_string())
    print("home margin mean/std:", round(data["home_margin"].mean(), 2), round(data["home_margin"].std(), 2))
    print(sorted(data["Home Team"].unique()))
