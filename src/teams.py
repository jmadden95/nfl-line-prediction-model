"""The 32 NFL franchises: canonical names, aliases, home coords, timezone, dome.

Canonical name = the current full name as used by aussportsbetting, ESPN and
The Odds API ("Kansas City Chiefs"). Historic/relocated names map to the same
franchise so form/Elo history carries across a move, while HISTORIC_HOME keeps
the OLD city's coordinates for travel distance in those seasons.
"""
from __future__ import annotations

import numpy as np

# name: (lat, lon, utc_offset_hours_standard, division, abbr, dome)
# dome: True for fixed/retractable-roof stadiums (default roof state). nflverse's
# per-game `roof` overrides this for historical rows.
TEAMS = {
    "Arizona Cardinals":      (33.528, -112.263, -7, "NFC West",  "ARI", True),
    "Atlanta Falcons":        (33.755,  -84.401, -5, "NFC South", "ATL", True),
    "Baltimore Ravens":       (39.278,  -76.623, -5, "AFC North", "BAL", False),
    "Buffalo Bills":          (42.774,  -78.787, -5, "AFC East",  "BUF", False),
    "Carolina Panthers":      (35.226,  -80.853, -5, "NFC South", "CAR", False),
    "Chicago Bears":          (41.862,  -87.617, -6, "NFC North", "CHI", False),
    "Cincinnati Bengals":     (39.095,  -84.516, -5, "AFC North", "CIN", False),
    "Cleveland Browns":       (41.506,  -81.700, -5, "AFC North", "CLE", False),
    "Dallas Cowboys":         (32.748,  -97.093, -6, "NFC East",  "DAL", True),
    "Denver Broncos":         (39.744, -105.020, -7, "AFC West",  "DEN", False),
    "Detroit Lions":          (42.340,  -83.046, -5, "NFC North", "DET", True),
    "Green Bay Packers":      (44.501,  -88.062, -6, "NFC North", "GB",  False),
    "Houston Texans":         (29.685,  -95.411, -6, "AFC South", "HOU", True),
    "Indianapolis Colts":     (39.760,  -86.164, -5, "AFC South", "IND", True),
    "Jacksonville Jaguars":   (30.324,  -81.637, -5, "AFC South", "JAX", False),
    "Kansas City Chiefs":     (39.049,  -94.484, -6, "AFC West",  "KC",  False),
    "Las Vegas Raiders":      (36.091, -115.184, -8, "AFC West",  "LV",  True),
    "Los Angeles Chargers":   (33.953, -118.339, -8, "AFC West",  "LAC", True),
    "Los Angeles Rams":       (33.953, -118.339, -8, "NFC West",  "LA",  True),
    "Miami Dolphins":         (25.958,  -80.239, -5, "AFC East",  "MIA", False),
    "Minnesota Vikings":      (44.974,  -93.258, -6, "NFC North", "MIN", True),
    "New England Patriots":   (42.091,  -71.264, -5, "AFC East",  "NE",  False),
    "New Orleans Saints":     (29.951,  -90.081, -6, "NFC South", "NO",  True),
    "New York Giants":        (40.813,  -74.074, -5, "NFC East",  "NYG", False),
    "New York Jets":          (40.813,  -74.074, -5, "AFC East",  "NYJ", False),
    "Philadelphia Eagles":    (39.901,  -75.168, -5, "NFC East",  "PHI", False),
    "Pittsburgh Steelers":    (40.447,  -80.016, -5, "AFC North", "PIT", False),
    "San Francisco 49ers":    (37.403, -121.970, -8, "NFC West",  "SF",  False),
    "Seattle Seahawks":       (47.595, -122.332, -8, "NFC West",  "SEA", False),
    "Tampa Bay Buccaneers":   (27.976,  -82.503, -5, "NFC South", "TB",  False),
    "Tennessee Titans":       (36.166,  -86.771, -6, "AFC South", "TEN", False),
    "Washington Commanders":  (38.908,  -76.864, -5, "NFC East",  "WAS", False),
}

# Historic / alternate spellings -> canonical franchise.
TEAM_NAME_FIXES = {
    "Washington Redskins": "Washington Commanders",
    "Washington Football Team": "Washington Commanders",
    "Washington": "Washington Commanders",
    "Oakland Raiders": "Las Vegas Raiders",
    "San Diego Chargers": "Los Angeles Chargers",
    "St. Louis Rams": "Los Angeles Rams",
    "St Louis Rams": "Los Angeles Rams",
    "LA Rams": "Los Angeles Rams",
    "LA Chargers": "Los Angeles Chargers",
    "NY Giants": "New York Giants",
    "NY Jets": "New York Jets",
    "Jacksonville Jaguars ": "Jacksonville Jaguars",
}

# Old-city coordinates for relocated franchises (raw name -> lat, lon, tz).
HISTORIC_HOME = {
    "Oakland Raiders": (37.752, -122.201, -8),
    "San Diego Chargers": (32.783, -117.120, -8),
    "St. Louis Rams": (38.633, -90.188, -6),
    "St Louis Rams": (38.633, -90.188, -6),
}

# nflverse abbreviations (incl. historic) -> canonical name.
ABBR_TO_NAME = {v[4]: k for k, v in TEAMS.items()}
ABBR_TO_NAME.update({"OAK": "Las Vegas Raiders", "SD": "Los Angeles Chargers",
                     "STL": "Los Angeles Rams", "LAR": "Los Angeles Rams",
                     "WSH": "Washington Commanders", "JAC": "Jacksonville Jaguars"})
NAME_TO_ABBR = {k: v[4] for k, v in TEAMS.items()}

TEAM_HOME = {k: (v[0], v[1]) for k, v in TEAMS.items()}
TEAM_TZ = {k: v[2] for k, v in TEAMS.items()}
TEAM_DIV = {k: v[3] for k, v in TEAMS.items()}
TEAM_DOME = {k: v[5] for k, v in TEAMS.items()}


def canon_team(name) -> str:
    """Map any spelling (historic, abbreviation, ESPN/Odds API) to canonical."""
    if name is None:
        return name
    s = str(name).strip()
    if s in TEAMS:
        return s
    if s in TEAM_NAME_FIXES:
        return TEAM_NAME_FIXES[s]
    if s.upper() in ABBR_TO_NAME:
        return ABBR_TO_NAME[s.upper()]
    # loose: match on the nickname (last word), e.g. "LA Rams", "Commanders"
    last = s.split()[-1].lower() if s.split() else ""
    for k in TEAMS:
        if k.split()[-1].lower() == last:
            # ambiguous nicknames (Giants/Jets share NY) are fine — nicknames are unique in the NFL
            return k
    return s


def short(name: str) -> str:
    """'Kansas City Chiefs' -> 'Chiefs'."""
    return str(name).split()[-1] if name else ""


def home_coords(raw_name: str, date=None):
    """(lat, lon) of a team's home at the time — old city for relocated names."""
    if raw_name in HISTORIC_HOME:
        h = HISTORIC_HOME[raw_name]
        return (h[0], h[1])
    return TEAM_HOME.get(canon_team(raw_name), (np.nan, np.nan))


def home_tz(raw_name: str) -> float:
    if raw_name in HISTORIC_HOME:
        return HISTORIC_HOME[raw_name][2]
    return TEAM_TZ.get(canon_team(raw_name), np.nan)


def haversine_km(lat1, lon1, lat2, lon2):
    """Great-circle distance (km) between two (lat, lon) points; vectorised."""
    r = 6371.0
    p1, p2 = np.radians(lat1), np.radians(lat2)
    dphi = p2 - p1
    dl = np.radians(lon2) - np.radians(lon1)
    a = np.sin(dphi / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dl / 2) ** 2
    return 2 * r * np.arcsin(np.sqrt(a))
