"""Central configuration for the NFL line model.

Sibling of the NRL line predictor (/docker/nrl-line-prediction-model), rebuilt
for the NFL. Same philosophy: predict the home margin with leakage-free
features, blend toward the market, and only flag a bet when the model clears
the vig. The extra angle here is the *market-structure* test — do Australian
books (Sportsbet et al) lag the global sharp line (Pinnacle) on an overseas
sport? We capture both on every poll so that can be measured, not assumed.
"""
from pathlib import Path
import os

from dotenv import load_dotenv

load_dotenv()

# --- Paths -----------------------------------------------------------------
ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
# aussportsbetting.com NFL workbook (same publisher/format as the NRL one).
# https://www.aussportsbetting.com/data/historical-nfl-results-and-odds-data/
RAW_XLSX = DATA_DIR / "nfl_results_and_odds.xlsx"
RAW_XLSX_URL = "https://www.aussportsbetting.com/historical_data/nfl.xlsx"

# nflverse open data: schedule + results + closing lines + rest days + roof +
# STARTING QB per game (lets the model LEARN what a backup QB costs).
NFLVERSE_GAMES_URL = "https://github.com/nflverse/nfldata/raw/master/data/games.csv"
NFLVERSE_GAMES_CSV = DATA_DIR / "nflverse_games.csv"
# nflverse official injury reports (practice + game status), per season.
NFLVERSE_INJURIES_URL = ("https://github.com/nflverse/nflverse-data/releases/download/"
                         "injuries/injuries_{season}.csv")

# ESPN public JSON (no key). NOTE: ESPN returns 403 to a *browser* User-Agent
# from this host but 200 to a plain/python one — do NOT spoof a browser UA.
ESPN_INJURIES_URL = "https://site.api.espn.com/apis/site/v2/sports/football/nfl/injuries"
ESPN_SCOREBOARD_URL = "https://site.api.espn.com/apis/site/v2/sports/football/nfl/scoreboard"

# --- Time ------------------------------------------------------------------
LOCAL_TZ = "Australia/Sydney"       # display / "today" for the AU bettor

# --- The Odds API ----------------------------------------------------------
ODDS_API_KEY = os.getenv("ODDS_API_KEY")
ODDS_SPORT = "americanfootball_nfl"
# Credits per call = markets x regions. "au" = the Australian books we bet
# with; "eu" carries Pinnacle, the global sharp reference. 3 markets x 2
# regions = 6 credits/call — see deploy/nfl-line.cron for the monthly budget.
ODDS_REGIONS = os.getenv("ODDS_REGIONS", "au,eu")
ODDS_MARKETS = os.getenv("ODDS_MARKETS", "h2h,spreads,totals")
ODDS_CACHE_TTL = int(os.getenv("ODDS_CACHE_TTL", "43200"))   # 12h page cache
# The book whose line is "the AU line" (bet-with book). Any AU book is captured.
AU_BOOK = os.getenv("AU_BOOK", "sportsbet")
# Books we treat as the sharp reference (median of whichever are present).
SHARP_BOOKS = [b.strip() for b in os.getenv("SHARP_BOOKS", "pinnacle").split(",") if b.strip()]
# Odds API bookmaker keys that are Australian books (region "au").
AU_BOOK_KEYS = {"sportsbet", "tab", "tabtouch", "neds", "ladbrokes_au", "pointsbetau",
                "unibet", "betfair_ex_au", "bet365_au", "betr_au", "dabble_au",
                "playup", "boombet", "bluebet", "topsport", "betright"}

# --- ntfy alerts -----------------------------------------------------------
NTFY_URL = os.getenv("NTFY_URL", "").rstrip("/")
NTFY_TOPIC = os.getenv("NTFY_TOPIC", "")
NTFY_TOKEN = os.getenv("NTFY_TOKEN", "")
# NFL lines move in half-points around key numbers (3, 7); 1.5 is a real move.
ALERT_MOVE_THRESHOLD = float(os.getenv("ALERT_MOVE_THRESHOLD", "1.5"))
# An AU book trailing the sharp line by this much is the "soft line" signal.
SHARP_LAG_MIN = float(os.getenv("SHARP_LAG_MIN", "1.0"))

# --- Player outs -----------------------------------------------------------
# Expected home-margin cost (points) when a STARTER at this position is ruled
# out, before stacking. Rough priors from public spread-impact research: a
# starting QB is worth several points; everyone else is 0.3-1.2. QB is the one
# that matters — and it is also a LEARNED feature (backup_qb_diff) trained on
# nflverse starting-QB history, so a QB out is applied through the model's
# fitted coefficient rather than this prior (see features.add_player_outs).
POSITION_POINTS = {
    "QB": 5.0,
    "T": 1.0, "OT": 1.0, "G": 0.6, "OG": 0.6, "C": 0.7, "OL": 0.7,
    "WR": 1.0, "TE": 0.7, "RB": 0.7, "FB": 0.2,
    "DE": 0.9, "EDGE": 0.9, "OLB": 0.8, "DT": 0.6, "NT": 0.5, "DL": 0.6,
    "LB": 0.6, "ILB": 0.6, "MLB": 0.6,
    "CB": 0.8, "S": 0.6, "FS": 0.6, "SS": 0.6, "DB": 0.6,
    "K": 0.5, "PK": 0.5, "P": 0.2, "LS": 0.1,
}
# Injury statuses that count as OUT for the upcoming game.
OUT_STATUSES = {"out", "injured reserve", "ir", "doubtful", "pup", "suspended", "nfi"}
# Diminishing returns on stacked outs (biggest counts full, next x decay ...).
OUT_STACK_DECAY = 0.7
# Cap the net non-QB out adjustment (pts). No NFL line moves 6 pts on non-QBs.
OUT_TEAM_CAP = 6.0
# A long-standing out is already in the Elo/form; fade its impact to zero over
# this many weeks since it was entered (NRL: 3; NFL plays weekly so 3 = 3 games).
ELO_ABSORB_WEEKS = 3.0
# Position weights only apply if we believe the player is a starter. Without a
# depth chart we assume an ESPN-listed Out player is a starter with this
# probability; the manual form can override points per player.
NON_QB_STARTER_PROB = 0.7

# --- Modelling constants ---------------------------------------------------
MARKET_BLEND = 0.5          # displayed line = 0.5*model + 0.5*AU line (as NRL)
SIGNAL_EDGE = 2.0           # |blended edge| (pts) to flag a handicap bet; re-sweep in backtest
TOTAL_SIGNAL = 3.0          # |model total - AU total| (pts) to flag an over/under
TOTAL_LAG_MIN = 1.5         # AU total vs sharp total gap (pts) that flags a soft total
RANDOM_SEED = 42
TEST_SEASONS = [2025]       # NFL season = the year it STARTS (2025 = Sep 2025-Feb 2026)
TRAIN_FROM_SEASON = 2008    # 2 warm-up seasons for form/Elo before training rows

ELO_K = 20.0                # FiveThirtyEight-style NFL Elo
ELO_SEASON_REGRESS = 0.67   # carry 2/3 of (rating-1500) into a new season
MARGIN_STD_DEV = 14.6       # std of NFL home margins 2006-2026 (5447 games)
