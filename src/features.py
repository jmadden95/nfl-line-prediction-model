"""Feature engineering — every feature uses only information known BEFORE kickoff.

Ported from the NRL model and adapted to the NFL:
  form_diff          rolling avg margin (last 5) home - away
  score_form_diff    rolling attack/defence differential
  rest_diff          days since last game (bye weeks, Thursday short weeks)
  travel_diff_km     away travel distance minus home travel (0 unless neutral)
  tz_shift           body-clock shift: home_tz - away_tz (+ = away team went east)
  h2h_home_margin    last-4 meetings, from the home team's view
  elo_diff           pre-game FiveThirtyEight-style Elo (MOV-weighted, season regress)
  div_game           divisional rivalry (tighter, lower-scoring)
  is_dome            fixed/retractable roof
  is_playoff         postseason
  backup_qb_diff     (away backup QB starting) - (home backup QB starting), from
                     nflverse starting-QB history -> LEARNED, unlike the NRL outs
  bookie_margin      the AU book's opening line as a home margin (benchmark + blend)

player_out_diff (non-QB outs from the injury feed / manual form) is NOT a
training feature (zero across history); it is added to the prediction, exactly
as in the NRL model. QB outs instead flip backup_qb_* on the upcoming row.

    python -m src.features     # leakage sanity check
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from config import (ELO_K, ELO_SEASON_REGRESS, ELO_ABSORB_WEEKS, OUT_STACK_DECAY,
                    OUT_TEAM_CAP, LOCAL_TZ)
from src.teams import (TEAM_DIV, TEAM_DOME, canon_team, home_coords, home_tz,
                       haversine_km)


def _explode(df: pd.DataFrame) -> pd.DataFrame:
    """Two rows per game (one per team) so a team's games sit together in order."""
    home = pd.DataFrame({"match_id": df.index, "date": df["Date"], "team": df["Home Team"],
                         "team_margin": df["home_margin"], "scored": df["Home Score"],
                         "conceded": df["Away Score"], "side": "home"})
    away = pd.DataFrame({"match_id": df.index, "date": df["Date"], "team": df["Away Team"],
                         "team_margin": -df["home_margin"], "scored": df["Away Score"],
                         "conceded": df["Home Score"], "side": "away"})
    long = pd.concat([home, away], ignore_index=True)
    return long.sort_values(["team", "date", "match_id"]).reset_index(drop=True)


def _attach(df, long, col, out_home, out_away):
    h = long[long["side"] == "home"].set_index("match_id")[col]
    a = long[long["side"] == "away"].set_index("match_id")[col]
    df[out_home] = h.reindex(df.index).to_numpy()
    df[out_away] = a.reindex(df.index).to_numpy()
    return df


def add_recent_form(df: pd.DataFrame, n: int = 5) -> pd.DataFrame:
    df = df.copy()
    long = _explode(df)
    g = long.groupby("team")
    long["form"] = g["team_margin"].transform(lambda s: s.shift(1).rolling(n, min_periods=1).mean())
    df = _attach(df, long, "form", "home_form", "away_form")
    df["form_diff"] = df["home_form"] - df["away_form"]
    return df


def add_scoring_form(df: pd.DataFrame, n: int = 5) -> pd.DataFrame:
    df = df.copy()
    long = _explode(df)
    g = long.groupby("team")
    long["attack"] = g["scored"].transform(lambda s: s.shift(1).rolling(n, min_periods=1).mean())
    long["defense"] = g["conceded"].transform(lambda s: s.shift(1).rolling(n, min_periods=1).mean())
    df = _attach(df, long, "attack", "home_attack", "away_attack")
    df = _attach(df, long, "defense", "home_defense", "away_defense")
    df["score_form_diff"] = ((df["home_attack"] - df["home_defense"])
                             - (df["away_attack"] - df["away_defense"]))
    return df


def add_rest_days(df: pd.DataFrame, cap: int = 21) -> pd.DataFrame:
    """Days since each team's previous game, capped (offseason -> cap)."""
    df = df.copy()
    long = _explode(df)
    long["rest"] = long.groupby("team")["date"].diff().dt.days.clip(upper=cap).fillna(cap)
    df = _attach(df, long, "rest", "home_rest_days", "away_rest_days")
    df["rest_diff"] = df["home_rest_days"] - df["away_rest_days"]
    return df


def add_travel(df: pd.DataFrame) -> pd.DataFrame:
    """Away team's trip to the home stadium (km) and the body-clock shift."""
    df = df.copy()
    rh = df.get("raw_home", df["Home Team"]).fillna(df["Home Team"])
    ra = df.get("raw_away", df["Away Team"]).fillna(df["Away Team"])
    hc = np.array([home_coords(x) for x in rh], dtype=float)
    ac = np.array([home_coords(x) for x in ra], dtype=float)
    km = haversine_km(ac[:, 0], ac[:, 1], hc[:, 0], hc[:, 1])
    neutral = df.get("neutral", pd.Series(0.0, index=df.index)).fillna(0).to_numpy() > 0
    df["away_travel_km"] = np.where(neutral, 0.0, km)      # unknown neutral venue -> no edge
    df["home_travel_km"] = 0.0
    df["travel_diff_km"] = df["away_travel_km"] - df["home_travel_km"]
    htz = np.array([home_tz(x) for x in rh], dtype=float)
    atz = np.array([home_tz(x) for x in ra], dtype=float)
    df["tz_shift"] = np.where(neutral, 0.0, htz - atz)
    return df


def add_h2h(df: pd.DataFrame, k: int = 4) -> pd.DataFrame:
    df = df.copy()
    pair = df.apply(lambda r: tuple(sorted([r["Home Team"], r["Away Team"]])), axis=1)
    # margin from the perspective of the alphabetically-first team
    first_is_home = pair.map(lambda p: p[0]) == df["Home Team"]
    m = np.where(first_is_home, df["home_margin"], -df["home_margin"])
    tmp = pd.DataFrame({"pair": pair, "m": m}, index=df.index)
    prior = tmp.groupby("pair")["m"].transform(lambda s: s.shift(1).rolling(k, min_periods=1).mean())
    df["h2h_home_margin"] = np.where(first_is_home, prior, -prior)
    return df


def add_elo(df: pd.DataFrame, k: float = None, regress: float = None) -> pd.DataFrame:
    """Pre-game Elo for each team (never sees the current game's result)."""
    k = ELO_K if k is None else k
    regress = ELO_SEASON_REGRESS if regress is None else regress
    df = df.copy()
    rating, last_season = {}, {}
    home_elo, away_elo = np.full(len(df), 1500.0), np.full(len(df), 1500.0)
    seasons = df["season"].to_numpy()
    for i, (h, a, s, m) in enumerate(zip(df["Home Team"], df["Away Team"], seasons, df["home_margin"])):
        for t in (h, a):
            rating.setdefault(t, 1500.0)
            if last_season.get(t) is not None and s != last_season[t]:
                rating[t] = 1500.0 + regress * (rating[t] - 1500.0)
            last_season[t] = s
        rh, ra = rating[h], rating[a]
        home_elo[i], away_elo[i] = rh, ra
        if pd.isna(m):
            continue                                       # upcoming game: no update
        diff = rh - ra + 48.0                              # ~2 pts of home field (Elo units)
        p_home = 1.0 / (1.0 + 10 ** (-diff / 400.0))
        s_home = 1.0 if m > 0 else (0.5 if m == 0 else 0.0)
        winner_diff = diff if m > 0 else -diff
        mov = np.log(abs(m) + 1.0) * (2.2 / (winner_diff * 0.001 + 2.2))   # 538 MOV multiplier
        delta = k * mov * (s_home - p_home)
        rating[h] += delta
        rating[a] -= delta
    df["home_elo"], df["away_elo"] = home_elo, away_elo
    df["elo_diff"] = df["home_elo"] - df["away_elo"]
    return df


def add_bookie_line(df: pd.DataFrame) -> pd.DataFrame:
    """A home line of -3.5 means the book expects home to win by 3.5 -> +3.5."""
    df = df.copy()
    df["bookie_margin"] = -pd.to_numeric(df.get("Home Line Open"), errors="coerce")
    df["close_margin"] = -pd.to_numeric(df.get("Home Line Close"), errors="coerce")
    return df


def add_context(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    same_div = (df["Home Team"].map(TEAM_DIV) == df["Away Team"].map(TEAM_DIV)).astype(float)
    dv = pd.to_numeric(df.get("div_game"), errors="coerce")
    df["div_game"] = dv.fillna(same_div) if dv is not None else same_div
    roof = df.get("roof")
    dome_default = df["Home Team"].map(TEAM_DOME).astype(float)
    if roof is not None:
        r = roof.astype(str).str.lower()
        df["is_dome"] = np.where(r.isin(["dome", "closed"]), 1.0,
                                 np.where(r.isin(["outdoors", "open"]), 0.0, dome_default))
    else:
        df["is_dome"] = dome_default
    df["is_playoff"] = pd.to_numeric(df.get("is_playoff"), errors="coerce").fillna(0.0)
    return df


def add_backup_qb(df: pd.DataFrame) -> pd.DataFrame:
    """1 if the team's starting QB is NOT its season-primary starter — the mode
    of its PRIOR games THIS season. The season opener (no prior games) is never
    flagged: an off-season change of starter is a deliberate choice, not a
    backup, and counting it would dilute the learned injury effect. Computed
    from nflverse qb names; upcoming rows get it from the outs table in
    add_player_outs."""
    df = df.copy()
    long = pd.concat([
        pd.DataFrame({"match_id": df.index, "date": df["Date"], "season": df["season"],
                      "team": df["Home Team"], "qb": df.get("home_qb_name"), "side": "home"}),
        pd.DataFrame({"match_id": df.index, "date": df["Date"], "season": df["season"],
                      "team": df["Away Team"], "qb": df.get("away_qb_name"), "side": "away"}),
    ], ignore_index=True).sort_values(["team", "date", "match_id"]).reset_index(drop=True)
    flags = np.zeros(len(long))
    primary_now = {}
    for team, g in long.groupby("team", sort=False):
        counts, cur_season = {}, None
        for idx, row in g.iterrows():
            if row["season"] != cur_season:
                counts, cur_season = {}, row["season"]
            qb = row["qb"] if isinstance(row["qb"], str) and row["qb"] else None
            primary = max(counts, key=counts.get) if counts else None
            if qb is not None and primary is not None and qb != primary:
                flags[idx] = 1.0
            if qb is not None:
                counts[qb] = counts.get(qb, 0) + 1
        primary_now[team] = max(counts, key=counts.get) if counts else None
    long["backup"] = flags
    df = _attach(df, long, "backup", "backup_qb_home", "backup_qb_away")
    df.attrs["primary_qb"] = primary_now          # team -> current primary starter
    return df


# --- player outs (live only) -------------------------------------------------
def _stack(points: list) -> float:
    pts = sorted([p for p in points if p and p > 0], reverse=True)
    return float(sum(p * (OUT_STACK_DECAY ** i) for i, p in enumerate(pts)))


def _absorb(entered, kickoff) -> float:
    """Fade an out's impact the longer it's been known (Elo already reflects it)."""
    try:
        weeks = (pd.Timestamp(kickoff) - pd.Timestamp(entered)).days / 7.0
    except Exception:  # noqa: BLE001
        return 1.0
    if ELO_ABSORB_WEEKS <= 0:
        return 1.0
    return float(np.clip(1.0 - max(weeks, 0.0) / ELO_ABSORB_WEEKS, 0.0, 1.0))


def active_outs(outs: pd.DataFrame, team: str, game_date) -> pd.DataFrame:
    """Outs for `team` whose window [entered, entered + weeks*7d) covers game_date."""
    if outs is None or not len(outs):
        return pd.DataFrame(columns=["player", "position", "points", "kind", "source", "entered", "weeks"])
    o = outs[outs["team"].map(canon_team) == canon_team(team)].copy()
    if not len(o):
        return o
    ent = pd.to_datetime(o["entered"], errors="coerce")
    wk = pd.to_numeric(o["weeks"], errors="coerce").fillna(1.0)
    gd = pd.Timestamp(game_date).normalize()
    ok = (ent.dt.normalize() <= gd) & (gd < ent.dt.normalize() + pd.to_timedelta(wk * 7, unit="D"))
    return o[ok.fillna(False)]


def add_player_outs(df: pd.DataFrame, outs: pd.DataFrame = None) -> pd.DataFrame:
    """Apply outs to UPCOMING rows: non-QB outs -> point adjustment (stacked,
    capped, absorbed); a starting-QB out flips backup_qb_* so the learned
    coefficient prices it. Historical rows are untouched (0)."""
    df = df.copy()
    for c in ["home_out_impact", "away_out_impact", "player_out_diff"]:
        df[c] = 0.0
    for c in ["backup_qb_home", "backup_qb_away"]:
        if c not in df.columns:
            df[c] = 0.0
        df[c] = df[c].fillna(0.0)
    df["home_outs"] = [[] for _ in range(len(df))]
    df["away_outs"] = [[] for _ in range(len(df))]
    up_mask = df.get("is_upcoming")
    if up_mask is None or not up_mask.any():
        df["backup_qb_diff"] = df["backup_qb_away"] - df["backup_qb_home"]
        return df
    if outs is None:
        try:
            from src import db
            outs = db.read_outs()
        except Exception:  # noqa: BLE001
            outs = None
    for i in df.index[up_mask.fillna(False)]:
        gd = df.at[i, "Date"]
        for side, team_col in (("home", "Home Team"), ("away", "Away Team")):
            act = active_outs(outs, df.at[i, team_col], gd)
            names, pts = [], []
            qb_out = False
            for _, o in act.iterrows():
                kind = str(o.get("kind") or "out")
                p = float(o.get("points") or 0.0) * _absorb(o.get("entered"), gd)
                pos = str(o.get("position") or "").upper()
                if pos == "QB" and kind == "out" and float(o.get("points") or 0) > 0:
                    qb_out = True
                    names.append({"player": o["player"], "position": pos, "points": None,
                                  "source": o.get("source"), "kind": kind, "qb": True})
                    continue
                signed = p if kind == "out" else -p            # "in" = returning, helps
                pts.append(signed)
                names.append({"player": o["player"], "position": pos, "points": round(signed, 2),
                              "source": o.get("source"), "kind": kind, "qb": False})
            outs_sum = min(_stack([p for p in pts if p > 0]), OUT_TEAM_CAP)
            ins_sum = min(_stack([-p for p in pts if p < 0]), OUT_TEAM_CAP)
            df.at[i, f"{side}_out_impact"] = outs_sum - ins_sum
            df.at[i, f"{side}_outs"] = names
            if qb_out:
                df.at[i, f"backup_qb_{side}"] = 1.0
    # positive favours home: the away side's holes minus the home side's
    df["player_out_diff"] = df["away_out_impact"] - df["home_out_impact"]
    df["backup_qb_diff"] = df["backup_qb_away"] - df["backup_qb_home"]
    return df


def build_features(df: pd.DataFrame, outs: pd.DataFrame = None) -> pd.DataFrame:
    df = add_recent_form(df)
    df = add_scoring_form(df)
    df = add_rest_days(df)
    df = add_travel(df)
    df = add_h2h(df)
    df = add_elo(df)
    df = add_bookie_line(df)
    df = add_context(df)
    df = add_backup_qb(df)
    df = add_player_outs(df, outs)
    return df


FEATURE_COLUMNS = [
    "form_diff", "score_form_diff", "rest_diff", "travel_diff_km", "tz_shift",
    "h2h_home_margin", "elo_diff", "div_game", "is_dome", "is_playoff",
    "backup_qb_diff", "bookie_margin",
]
LINE_FREE_COLUMNS = [c for c in FEATURE_COLUMNS if c != "bookie_margin"]


if __name__ == "__main__":
    from src.load_data import add_margin, load_raw
    df = build_features(add_margin(load_raw()))
    team = "Kansas City Chiefs"
    one = df[(df["Home Team"] == team) | (df["Away Team"] == team)].tail(8)
    cols = ["Date", "Home Team", "Away Team", "home_margin", "form_diff", "rest_diff",
            "elo_diff", "backup_qb_home", "backup_qb_away", "bookie_margin"]
    print(one[cols].to_string(index=False))
    tr = df.dropna(subset=["home_margin", "bookie_margin"])
    print("\nCorrelation with home_margin (train rows):")
    print(tr[FEATURE_COLUMNS + ["home_margin"]].corr()["home_margin"].round(3).to_string())
    print("\nbackup QB starts (home/away):", int(tr["backup_qb_home"].sum()), int(tr["backup_qb_away"].sum()))
