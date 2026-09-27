# NFL Line Predictor

Sibling of the NRL line model (`/docker/nrl-line-prediction-model`), rebuilt for
the NFL with one extra question baked in: **are Australian books soft on an
overseas sport?** Every poll captures the AU line (Sportsbet + every other AU
book) *and* the sharp line (Pinnacle), so the answer is measured, not assumed.

## What it does
- Predicts the home margin with a leakage-free linear model (form, rest, travel,
  body-clock, head-to-head, Elo, division, dome, playoff, **backup-QB**), blended
  50/50 toward the AU opening line. Flags a bet when |edge| >= `SIGNAL_EDGE`.
- Flags **AU soft lines**: any AU book whose handicap trails the sharp line (median of Pinnacle, FanDuel, DraftKings) by >=
  `SHARP_LAG_MIN` points (ntfy alert, high priority).
- Outs: ESPN injury feed (Out / Doubtful / IR) sized by position; a starting QB
  out goes through the model's *learned* backup-QB coefficient (from nflverse
  starting-QB history) instead of a guessed number. Manual overrides in the UI.
- Results auto-capture from ESPN (free), so the model keeps training itself;
  bets auto-settle with closing-line value.
- Walk-forward backtest + market-structure report in the UI.

## Inputs
| Input | Source | Cost |
|---|---|---|
| History 2006+ (results, open/close lines, totals). NB lines are Pinnacle 2014-18, bet365 2018-25, Betr (AU) Sep 2025+ | aussportsbetting.com `nfl.xlsx` -> `data/nfl_results_and_odds.xlsx` | free, manual download |
| Schedule, rest, roof, starting QBs, US closing spread | nflverse `games.csv` (auto-refreshed) | free |
| Live odds (AU + Pinnacle + FanDuel/DraftKings) | The Odds API `americanfootball_nfl`, regions au,eu,us | 9 credits per call |
| Injuries | ESPN public JSON | free (plain User-Agent only) |
| Results | ESPN scoreboard | free |

## Run
```bash
cp .env.example .env   # add ODDS_API_KEY (+ ntfy token)
docker compose up -d --build
./run.sh backtest      # walk-forward + market structure
./run.sh injuries      # sync outs from ESPN
./run.sh live          # print the board
```
Cron: `deploy/nfl-line.cron` -> `/etc/cron.d/nfl-line`. Wiki: BookStack Services book, page "NFL Line Predictor".
