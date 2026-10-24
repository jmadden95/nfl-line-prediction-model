"""Named bet strategies: two HEADLINE (bet buttons + alerts), the rest tracked in the background.

Every strategy is a function (game row, context) -> list of picks. Each pick is recorded
ONCE per game/strategy/side at the moment it first fires (pre-kickoff), with the line,
price and book at that moment, so every strategy builds an honest forward record whether
or not you bet it. strategy_scorecard() scores them: CLV vs the AU-consensus close, result,
ROI at the logged price.

Why these two lead (forward, 2026 wk2-7, pre-kickoff captures):
  key_number_dog  AU consensus has the dog at +3/+7 or less and an AU book offers +3.5/+7.5:
                  28 settled, 64% cover, +22.7% — 14% of NFL games land exactly on 3, 9% on 7.
  wind_under      outdoor, forecast >= 15 mph: historically 60% unders vs the open (n=417,
                  2012-25, both halves). Dormant so far in 2026 (no 15 mph game yet).
Background (no buttons): model_handicap, model_total, au_soft (Sportsbet vs Pinnacle),
  soft_book_outlier (non-Sportsbet AU book >= 1 off AU consensus — those covered ~60%),
  sportsbet_outlier (Sportsbet >= 1 off AU consensus — 42%, it LEADS the AU market).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from config import AU_BOOK_KEYS, LOCAL_TZ, SHARP_LAG_MIN
from src import db
from src.signals import handicap_signal, total_signal
from src.teams import short

KEY_NUMBERS = (3, 7)
WIND_UNDER_MPH = 15.0
MIN_PRICE = 1.80          # don't pay more than ~-125 for the half point
LEADER = "sportsbet"      # the AU book the others follow


def _f(x):
    try:
        v = float(x)
        return None if v != v else v
    except (TypeError, ValueError):
        return None


def au_books(books):
    return [b for b in (books or []) if b.get("region") == "au" and _f(b.get("sp_point")) is not None]


def au_consensus(books):
    pts = [float(b["sp_point"]) for b in au_books(books)]
    return (float(np.median(pts)), len(pts)) if pts else (None, 0)


def _pick(strategy, market, side, line, odds, book, note=""):
    return {"strategy": strategy, "market": market, "side": side, "line": float(line),
            "odds": _f(odds), "book": book, "note": note}


# --- headline -----------------------------------------------------------------
def key_number_dog(r, ctx):
    books = ctx["books"]
    cons, n = au_consensus(books)
    if cons is None or n < 3:
        return []
    out = []
    for side in ("home", "away"):
        sg = 1 if side == "home" else -1
        cons_dog = cons * sg                       # this side's consensus handicap (+ = dog)
        if cons_dog <= 0:
            continue
        for k in KEY_NUMBERS:
            if not (k - 1 < cons_dog <= k):
                continue
            offers = [(float(b["sp_point"]) * sg, _f(b.get("sp_home") if side == "home" else b.get("sp_away")), b["title"])
                      for b in au_books(books)]
            offers = [o for o in offers if o[0] >= k + 0.5 and (o[1] is None or o[1] >= MIN_PRICE)]
            if not offers:
                continue
            best = max(offers, key=lambda o: (o[0], o[1] or 0))
            out.append(_pick("key_number_dog", "hcap", side, best[0], best[1], best[2],
                             f"AU consensus {cons_dog:+.1f}, {best[2]} {best[0]:+.1f} crosses {k}"))
            break
    return out


def wind_under(r, ctx):
    if ctx.get("enclosed") is not False:
        return []
    wind = _f(ctx.get("wind"))
    if wind is None or wind < WIND_UNDER_MPH:
        return []
    tots = [(float(b["tot_point"]), _f(b.get("tot_under")), b["title"]) for b in (ctx["books"] or [])
            if b.get("region") == "au" and _f(b.get("tot_point")) is not None]
    if not tots:
        return []
    best = max(tots, key=lambda o: (o[0], o[1] or 0))       # highest total, best under price
    return [_pick("wind_under", "total", "under", best[0], best[1], best[2],
                  f"forecast {wind:.0f} mph at {ctx.get('venue') or 'venue'}")]


# --- background ---------------------------------------------------------------
def model_handicap(r, ctx):
    side = handicap_signal(ctx.get("edge"), ctx.get("bookie_margin"))
    ap = _f(ctx.get("au_point"))
    if not side or ap is None:
        return []
    line = ap if side == "home" else -ap
    odds = ctx.get("au_home_odds") if side == "home" else ctx.get("au_away_odds")
    return [_pick("model_handicap", "hcap", side, line, odds, ctx.get("au_book") or "Sportsbet", f"edge {float(ctx['edge']):+.1f}")]


def model_total(r, ctx):
    pt, at = _f(ctx.get("pred_total")), _f(ctx.get("au_total"))
    if pt is None or at is None:
        return []
    side = total_signal(pt - at)
    if not side:
        return []
    odds = ctx.get("au_over") if side == "over" else ctx.get("au_under")
    return [_pick("model_total", "total", side, at, odds, ctx.get("au_book") or "Sportsbet", f"model {pt:.1f}")]


def au_soft(r, ctx):
    ap, sp = _f(ctx.get("au_point")), _f(ctx.get("sharp_point"))
    if ap is None or sp is None or abs(ap - sp) < SHARP_LAG_MIN:
        return []
    side = "home" if ap - sp > 0 else "away"
    line = ap if side == "home" else -ap
    odds = ctx.get("au_home_odds") if side == "home" else ctx.get("au_away_odds")
    return [_pick("au_soft", "hcap", side, line, odds, ctx.get("au_book") or "Sportsbet", f"Pinnacle {sp:+.1f}")]


def _outlier(r, ctx, name, want_leader: bool):
    books = ctx["books"]
    cons, n = au_consensus(books)
    if cons is None or n < 4:
        return []
    out = []
    for b in au_books(books):
        is_leader = b.get("key") == LEADER
        if is_leader != want_leader:
            continue
        dev = float(b["sp_point"]) - cons
        if abs(dev) < 1.0:
            continue
        side = "home" if dev > 0 else "away"
        line = float(b["sp_point"]) if side == "home" else -float(b["sp_point"])
        odds = b.get("sp_home") if side == "home" else b.get("sp_away")
        out.append(_pick(name, "hcap", side, line, odds, b["title"], f"{dev:+.1f} vs AU consensus {cons:+.1f}"))
    return out


def soft_book_outlier(r, ctx):
    return _outlier(r, ctx, "soft_book_outlier", want_leader=False)


def sportsbet_outlier(r, ctx):
    return _outlier(r, ctx, "sportsbet_outlier", want_leader=True)


STRATEGIES = {
    "key_number_dog": {"fn": key_number_dog, "headline": True, "label": "Key-number dog",
                       "blurb": "An AU book gives the dog +3.5/+7.5 while the AU consensus sits at +3/+7 or less."},
    "wind_under": {"fn": wind_under, "headline": True, "label": "Wind under",
                   "blurb": "Outdoor game with 15+ mph forecast: take the highest AU total under."},
    "model_handicap": {"fn": model_handicap, "headline": False, "label": "Model handicap (fav ≥2 / dog ≥3.5)"},
    "model_total": {"fn": model_total, "headline": False, "label": "Model total (under ≥2 / over ≥3, wind-aware)"},
    "au_soft": {"fn": au_soft, "headline": False, "label": "Sportsbet trails Pinnacle ≥1"},
    "soft_book_outlier": {"fn": soft_book_outlier, "headline": False, "label": "Soft AU book ≥1 off AU consensus"},
    "sportsbet_outlier": {"fn": sportsbet_outlier, "headline": False, "label": "Sportsbet ≥1 off AU consensus"},
}


def evaluate(r, ctx) -> list:
    picks = []
    for name, meta in STRATEGIES.items():
        try:
            picks.extend(meta["fn"](r, ctx))
        except Exception:  # noqa: BLE001 - one strategy's bug must not break the board
            pass
    return picks


# --- recording --------------------------------------------------------------------
def _ensure(conn):
    conn.execute("CREATE TABLE IF NOT EXISTS strategy_picks(first_ts TEXT, game_date TEXT, home TEXT, away TEXT, "
                 "strategy TEXT, market TEXT, side TEXT, line REAL, odds REAL, book TEXT, note TEXT, "
                 "UNIQUE(game_date, home, away, strategy, side))")


def record(game_date, home, away, picks, kickoff_utc=None, ts=None) -> int:
    """Store first-flag picks (ignored if that game/strategy/side already exists). Pre-kickoff only."""
    if not picks:
        return 0
    if kickoff_utc is not None:
        try:
            if pd.Timestamp(kickoff_utc) <= pd.Timestamp.now(tz="UTC"):
                return 0
        except Exception:  # noqa: BLE001
            pass
    ts = ts or pd.Timestamp.now(tz=LOCAL_TZ).strftime("%Y-%m-%dT%H:%M:%S")
    conn = db.connect()
    n = 0
    try:
        _ensure(conn)
        for p in picks:
            cur = conn.execute("INSERT OR IGNORE INTO strategy_picks VALUES (?,?,?,?,?,?,?,?,?,?,?)",
                               (ts, str(pd.Timestamp(game_date).date()), home, away, p["strategy"], p["market"],
                                p["side"], p["line"], p["odds"], p["book"], p["note"]))
            n += cur.rowcount
        conn.commit()
    finally:
        conn.close()
    return n


def picks_df() -> pd.DataFrame:
    conn = db.connect()
    try:
        _ensure(conn)
        return pd.read_sql("SELECT * FROM strategy_picks", conn)
    finally:
        conn.close()


# --- scoring ----------------------------------------------------------------------
def _closes():
    """(gd, home, away) -> (AU consensus home point at last pre-KO capture, AU median total)."""
    bl = db.book_lines_df()
    if not len(bl):
        return {}
    bl = bl[bl["book"].isin(AU_BOOK_KEYS)]
    bl["cts"] = pd.to_datetime(bl["captured_ts"], format="ISO8601", errors="coerce")
    last = bl.groupby(["game_date", "home", "away"])["cts"].transform("max")
    bl = bl[bl["cts"] == last]
    g = bl.groupby(["game_date", "home", "away"]).agg(pt=("sp_point", "median"), tot=("tot_point", "median"))
    return {k: (row.pt, row.tot) for k, row in g.iterrows()}


def strategy_scorecard() -> list:
    from src.bets import _results_index
    picks = picks_df()
    if not len(picks):
        return []
    res = _results_index()
    closes = _closes()
    rows = []
    for _, p in picks.iterrows():
        k = (p["game_date"], p["home"], p["away"])
        c = closes.get(k)
        r = res.get(k)
        sg = 1 if p["side"] == "home" else -1
        clv = cover = np.nan
        if p["market"] == "hcap":
            if c and c[0] == c[0]:
                clv = p["line"] - c[0] * sg
            if r:
                cover = r[0] * sg + p["line"]
        else:
            over = p["side"] == "over"
            if c and c[1] == c[1]:
                clv = (c[1] - p["line"]) if over else (p["line"] - c[1])
            if r:
                cover = (r[1] - p["line"]) if over else (p["line"] - r[1])
        odds = p["odds"] if p["odds"] and p["odds"] == p["odds"] else 1.909
        profit = np.nan if cover != cover else (odds - 1 if cover > 0 else (-1.0 if cover < 0 else 0.0))
        rows.append({"strategy": p["strategy"], "clv": clv, "cover": cover, "profit": profit})
    d = pd.DataFrame(rows)
    out = []
    for name, meta in STRATEGIES.items():
        g = d[d["strategy"] == name]
        if not len(g):
            out.append({"strategy": name, "label": meta["label"], "headline": meta["headline"], "n": 0})
            continue
        dec = g.dropna(subset=["cover"]); dec = dec[dec["cover"] != 0]
        clv = g["clv"].dropna()
        out.append({"strategy": name, "label": meta["label"], "headline": meta["headline"], "n": int(len(g)),
                    "settled": int(len(dec)), "w": int((dec["cover"] > 0).sum()), "l": int((dec["cover"] < 0).sum()),
                    "cover": float((dec["cover"] > 0).mean()) if len(dec) else None,
                    "roi": float(dec["profit"].sum() / len(dec)) if len(dec) else None,
                    "clv_med": float(clv.median()) if len(clv) else None,
                    "beat": float((clv > 0).mean()) if len(clv) else None})
    return out


# --- backfill from captured history ------------------------------------------------
def backfill() -> int:
    """Replay every pre-kickoff capture chronologically and record each strategy's FIRST
    flag with the line/price at that moment. Uses book_lines (all books), predictions
    (model edge/total per day) and weather_forecast. Idempotent."""
    bl = db.book_lines_df()
    if not len(bl):
        return 0
    bl["cts"] = pd.to_datetime(bl["captured_ts"], format="ISO8601", errors="coerce")
    snaps = db.odds_snapshots_df()
    snaps["cts"] = pd.to_datetime(snaps["captured_ts"], format="ISO8601", errors="coerce")
    kos = snaps.groupby(["game_date", "home", "away"])["kickoff"].first().map(lambda x: pd.to_datetime(x, format="mixed", errors="coerce"))
    preds = db.predictions_df(); preds["cts"] = pd.to_datetime(preds["captured_ts"], format="ISO8601", errors="coerce")
    wx = db._df("select * from weather_forecast"); wx["cts"] = pd.to_datetime(wx["captured_ts"], format="ISO8601", errors="coerce")
    from src.weather import venue_for
    n = 0
    for (gd, h, a), g in bl.groupby(["game_date", "home", "away"]):
        ko = kos.get((gd, h, a))
        if ko is None or ko != ko:
            continue
        v = venue_for(h, a, gd)
        for cts, cap in g[g["cts"] < ko].sort_values("cts").groupby("cts"):
            books = [{"key": b.book, "title": b.title, "region": b.region, "sharp": b.book == "pinnacle",
                      "sp_point": b.sp_point, "sp_home": b.sp_home, "sp_away": b.sp_away,
                      "tot_point": b.tot_point, "tot_over": b.tot_over, "tot_under": b.tot_under} for b in cap.itertuples()]
            sb = next((b for b in books if b["key"] == "sportsbet"), None)
            pin = next((b for b in books if b["key"] == "pinnacle"), None)
            pr = preds[(preds.game_date == gd) & (preds.home == h) & (preds.away == a) & (preds.cts <= cts)].sort_values("cts")
            pr = pr.iloc[-1] if len(pr) else None
            w = wx[(wx.game_date == gd) & (wx.home == h) & (wx.away == a) & (wx.cts <= cts)].sort_values("cts")
            w = w.iloc[-1] if len(w) else None
            ctx = {"books": books, "au_point": sb and sb["sp_point"], "au_home_odds": sb and sb["sp_home"],
                   "au_away_odds": sb and sb["sp_away"], "au_total": sb and sb["tot_point"],
                   "au_over": sb and sb["tot_over"], "au_under": sb and sb["tot_under"], "au_book": sb and sb["title"],
                   "sharp_point": pin and pin["sp_point"], "bookie_margin": -sb["sp_point"] if sb and sb["sp_point"] is not None else None,
                   "edge": pr["edge"] if pr is not None else None, "pred_total": pr.get("pred_total") if pr is not None else None,
                   "enclosed": v.get("enclosed"), "venue": v.get("venue"),
                   "wind": (0.0 if v.get("enclosed") else (w["wind_mph"] if w is not None else None))}
            n += record(gd, h, a, evaluate(None, ctx), ts=cts.strftime("%Y-%m-%dT%H:%M:%S"))
    return n


if __name__ == "__main__":
    import sys
    db.init_db()
    if "--backfill" in sys.argv:
        print(f"backfilled {backfill()} first-flag picks")
    for s in strategy_scorecard():
        if s["n"]:
            print(f"{'*' if s['headline'] else ' '} {s['label']:<48} n={s['n']:3d} settled={s['settled']:3d} {s['w']}-{s['l']} "
                  f"cover {s['cover']:.0%} ROI {s['roi']:+.1%} | CLV med {s['clv_med']:+.2f} beat {s['beat']:.0%}")
        else:
            print(f"{'*' if s['headline'] else ' '} {s['label']:<48} n=0")


# --- live board integration -------------------------------------------------------
def ctx_from_row(r) -> dict:
    """Build a strategy context from one live predict_upcoming row."""
    g = lambda k: r.get(k) if hasattr(r, "get") else getattr(r, k, None)
    enc = g("enclosed")
    if isinstance(enc, float) and enc != enc:
        enc = None
    return {"books": g("books") or [], "au_point": g("au_point"), "au_home_odds": g("au_home_odds"),
            "au_away_odds": g("au_away_odds"), "au_total": g("au_total"), "au_over": g("au_over"),
            "au_under": g("au_under"), "au_book": g("au_book"), "sharp_point": g("sharp_point"),
            "bookie_margin": g("bookie_margin"), "edge": g("edge"), "pred_total": g("pred_total"),
            "enclosed": (bool(enc) if enc is not None else None), "venue": g("venue"), "wind": g("wind")}


def apply_to_board(up, do_record: bool = True) -> dict:
    """(game_date, home, away) -> picks for every upcoming row; records first flags."""
    out = {}
    if up is None or not len(up):
        return out
    for _, r in up.iterrows():
        picks = evaluate(r, ctx_from_row(r))
        gd = str(pd.Timestamp(r["Date"]).date())
        out[(gd, r["Home Team"], r["Away Team"])] = picks
        if do_record and picks:
            try:
                record(gd, r["Home Team"], r["Away Team"], picks, kickoff_utc=r.get("kickoff_utc"))
            except Exception:  # noqa: BLE001
                pass
    return out


def alert_headline(up, picks_by_game: dict) -> int:
    """ntfy for HEADLINE strategy picks only, once per game/strategy/side."""
    from src.notify import push
    sent = 0
    for (gd, h, a), picks in picks_by_game.items():
        for p in picks:
            if not STRATEGIES[p["strategy"]]["headline"]:
                continue
            key = f"strat|{gd}|{h}|{a}|{p['strategy']}|{p['side']}"
            if db.was_alerted(key):
                continue
            team = {"home": short(h), "away": short(a)}.get(p["side"], p["side"].upper())
            line = f"{p['line']:+.1f}" if p["market"] == "hcap" else f"{p['line']:.1f}"
            price = f" @ {p['odds']:.2f}" if p.get("odds") else ""
            if push(f"NFL {STRATEGIES[p['strategy']]['label']}: {team} {line}",
                    f"{a} @ {h} ({gd}) — {team} {line}{price} at {p['book']}. {p['note']}", tags="dart"):
                db.mark_alerted(key); sent += 1
    return sent
