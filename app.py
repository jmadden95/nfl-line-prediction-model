"""NFL Line Predictor — web UI. Sibling of the NRL app; runs behind Caddy.

Local dev:   python app.py          In Docker: gunicorn app:app
"""
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
from flask import Flask, jsonify, redirect, render_template_string, request, url_for

from config import (DATA_DIR, LOCAL_TZ, MARKET_BLEND, NFLVERSE_GAMES_CSV, NTFY_TOPIC, NTFY_URL,
                    ODDS_API_KEY, ODDS_MARKETS, ODDS_REGIONS, POSITION_POINTS, RAW_XLSX,
                    SHARP_LAG_MIN, SIGNAL_EDGE, TOTAL_LAG_MIN, TOTAL_SIGNAL, FAV_EDGE, DOG_EDGE,
                    TOTAL_UNDER_EDGE, TOTAL_OVER_EDGE)
from src.signals import handicap_signal, total_signal
from src import db
from src import bankroll as bk
from src.bets import settle_bets, signal_scorecards, tally
from src.live import cache_age_seconds, fetch_odds, mock_events, predict_upcoming
from src.teams import TEAMS, canon_team, short
from src.win_prob import cover_prob

app = Flask(__name__)
SHOW_DAYS = 8
MARGIN_COVER_STD = 20.0      # betting sigma for edge -> cover prob (wider than raw 14.6; NRL used 22 on 18.4)
TOTAL_COVER_STD = 18.0       # same idea for totals (NFL totals residual sd ~13.5; widened for edge noise)

try:
    db.init_db()
except Exception:  # noqa: BLE001
    pass


def _now():
    return pd.Timestamp.now(tz=LOCAL_TZ).tz_localize(None)


def _f(x, nd=1, default="–"):
    try:
        if x is None or (isinstance(x, float) and math.isnan(x)) or pd.isna(x):
            return default
        return f"{float(x):.{nd}f}"
    except Exception:  # noqa: BLE001
        return default


def _signed(x, nd=1):
    try:
        if pd.isna(x):
            return "–"
        return f"{float(x):+.{nd}f}"
    except Exception:  # noqa: BLE001
        return "–"


def _hcap(home, away, home_point):
    """Home handicap point -> 'Chiefs -3.5' style phrase."""
    if home_point is None or pd.isna(home_point):
        return "–"
    hp = float(home_point)
    return f"{short(home)} {hp:+.1f}" if hp <= 0 else f"{short(away)} {-hp:+.1f}"


def _pick(r):
    e = r.get("edge")
    if e is None or pd.isna(e) or pd.isna(r.get("au_point")):
        return None
    ap = float(r["au_point"])
    side = handicap_signal(e, r.get("bookie_margin"))
    if side == "home":
        return {"team": short(r["Home Team"]), "line": ap, "side": "home", "edge": float(e),
                "odds": r.get("au_home_odds")}
    if side == "away":
        return {"team": short(r["Away Team"]), "line": -ap, "side": "away", "edge": float(-e),
                "odds": r.get("au_away_odds")}
    return None


def _sharp_call(r):
    g = r.get("au_vs_sharp")
    if g is None or pd.isna(g) or abs(g) < SHARP_LAG_MIN:
        return None
    ap = float(r["au_point"])
    if g > 0:
        return {"team": short(r["Home Team"]), "line": ap, "gap": float(g)}
    return {"team": short(r["Away Team"]), "line": -ap, "gap": float(-g)}


def _au_shop(books, side_home: bool):
    """Every AU book's handicap for one side, best (most points, then price) first."""
    rows = []
    for b in books or []:
        if b.get("region") != "au" or b.get("sp_point") is None:
            continue
        pt = float(b["sp_point"]) if side_home else -float(b["sp_point"])
        price = b.get("sp_home") if side_home else b.get("sp_away")
        rows.append({"book": b["title"], "point": pt, "price": price})
    return sorted(rows, key=lambda x: (-x["point"], -(x["price"] or 0)))


def _au_shop_totals(books, over: bool):
    """AU books' total for one side; best first (over: lowest line then price; under: highest)."""
    rows = []
    for b in books or []:
        if b.get("region") != "au" or b.get("tot_point") is None:
            continue
        rows.append({"book": b["title"], "point": float(b["tot_point"]),
                     "price": b.get("tot_over") if over else b.get("tot_under")})
    return sorted(rows, key=lambda x: ((x["point"] if over else -x["point"]), -(x["price"] or 0)))


def _totals_shop(books):
    """Every AU book's total (+ Pinnacle for reference). Best OVER = lowest line then
    best price; best UNDER = highest line then best price."""
    rows = []
    for b in books or []:
        if b.get("tot_point") is None or not (b.get("region") == "au" or b.get("sharp")):
            continue
        rows.append({"book": b["title"], "point": float(b["tot_point"]), "over": b.get("tot_over"),
                     "under": b.get("tot_under"), "sharp": bool(b.get("sharp")) and b.get("region") != "au"})
    au = [x for x in rows if not x["sharp"]]
    if not au:
        return None
    bo = min(au, key=lambda x: (x["point"], -(x["over"] or 0)))
    bu = max(au, key=lambda x: (x["point"], x["under"] or 0))
    for x in rows:
        x["best_over"] = x is bo
        x["best_under"] = x is bu
    pts = [x["point"] for x in au]
    return {"rows": sorted(rows, key=lambda x: (x["sharp"], x["point"], x["book"])), "best_over": bo, "best_under": bu,
            "lo": min(pts), "hi": max(pts), "n": len(au)}


def _line_history(gd, home, away):
    snaps = db.odds_snapshots_df()
    if not len(snaps):
        return []
    g = snaps[(snaps["game_date"] == gd) & (snaps["home"] == home) & (snaps["away"] == away)]
    return [{"phase": s["phase"], "ts": str(s["captured_ts"])[5:16].replace("T", " "),
             "au": s["au_point"], "sharp": s["sharp_point"]} for _, s in g.sort_values("captured_ts").iterrows()]


def _status():
    st = {}
    try:
        from src.load_data import load_raw
        m = load_raw()
        done = m.dropna(subset=["Home Score"])
        st["games"] = len(done)
        st["last_result"] = f"{done['Date'].max():%Y-%m-%d}"
        st["nflverse_match"] = f"{m['week'].notna().mean():.0%}"
    except Exception as ex:  # noqa: BLE001
        st["games"], st["last_result"], st["nflverse_match"] = 0, f"no data ({ex})", "–"
    age = cache_age_seconds()
    st["odds_age"] = "no cache" if age is None else (f"{age/3600:.1f}h old" if age > 3600 else f"{age/60:.0f}m old")
    st["api"] = db.api_usage()
    st["injury"] = db.get_json("injury_status", {})
    st["ntfy"] = bool(NTFY_URL and NTFY_TOPIC)
    st["nflverse_age"] = (f"{(pd.Timestamp.now().timestamp() - NFLVERSE_GAMES_CSV.stat().st_mtime)/86400:.1f}d"
                          if NFLVERSE_GAMES_CSV.exists() else "missing")
    st["xlsx"] = RAW_XLSX.exists()
    st["credits_per_call"] = len(ODDS_MARKETS.split(",")) * len(ODDS_REGIONS.split(","))
    return st


def _coeffs():
    c = db.model_coeffs_df()
    if not len(c):
        return []
    latest = c[c["captured_date"] == c["captured_date"].max()]
    return [(r["feature"], r["coef"]) for _, r in latest.sort_values("coef", key=abs, ascending=False).iterrows()]


def get_board(force_fresh: bool = False):
    warn = None
    events = []
    if not ODDS_API_KEY:
        events, warn = mock_events(), "No ODDS_API_KEY — showing a MOCK fixture."
    else:
        try:
            events = fetch_odds(use_cache=not force_fresh)
        except Exception as ex:  # noqa: BLE001
            warn = f"Odds fetch failed: {ex}"
    if not events:
        return [], warn, set()
    up = predict_upcoming(events)
    if not len(up):
        return [], warn or "No upcoming games in the odds feed.", set()
    try:
        from src.weather import capture_forecast
        if capture_forecast(events):          # throttled (3h per game); free API
            up = predict_upcoming(events)     # re-run so the fresh wind feeds the totals
    except Exception:  # noqa: BLE001
        pass
    # totals
    try:
        from src.features import build_features
        from src.load_data import add_margin, load_raw
        from src.totals import predict_totals
        feat_hist = build_features(add_margin(load_raw()))
        up["pred_total"] = predict_totals(feat_hist, up, up["au_total"])
    except Exception:  # noqa: BLE001
        up["pred_total"] = np.nan
    try:
        # Forward capture on view, throttled to once per game per day: the model's
        # numbers AND the AU-vs-sharp lines (this is the data the thesis is judged on).
        db.snapshot_predictions(up, phase="page")
        if db.snapshot_odds(events, phase="page"):
            db.snapshot_book_lines(events, phase="page")
    except Exception:  # noqa: BLE001
        pass
    horizon = _now() + pd.Timedelta(days=SHOW_DAYS)
    bank = None
    try:
        bank = bk.current()
    except Exception:  # noqa: BLE001
        bank = None
    bank_total = bank["total"] if bank else None
    # Pending bets by (home, away, side) so every pick can show "already bet".
    placed, my_bets = set(), {}
    try:
        _b = db.read_bets()
        for _, b in _b[~_b["result"].isin(["W", "L", "P"])].iterrows():
            side = str(b["side"]).lower()
            gk = (canon_team(b["home"]), canon_team(b["away"]))
            placed.add((*gk, side))
            ln = float(b["line"])
            lbl = (f"{side.upper()} {ln:g}" if side in ("over", "under")
                   else f"{short(b['home'] if side == 'home' else b['away'])} {ln:+g}")
            my_bets.setdefault(gk, []).append(lbl)
    except Exception:  # noqa: BLE001
        pass
    cards = []
    for _, r in up.iterrows():
        ko = pd.to_datetime(r["Kickoff"], errors="coerce")
        if ko is not None and not pd.isna(ko) and ko > horizon:
            continue
        pick = _pick(r)
        gd = str(pd.Timestamp(r["Date"]).date())
        t_edge = (r["pred_total"] - r["au_total"]) if pd.notna(r.get("au_total")) and pd.notna(r.get("pred_total")) else np.nan
        soft = _sharp_call(r)
        books = r.get("books")
        def _form(side, line, edge_pts, model):
            shop = _au_shop(books, side == "home")
            best = shop[0] if shop else None
            odds = (best["price"] if best and best["price"] else
                    (r.get("au_home_odds") if side == "home" else r.get("au_away_odds")))
            odds = float(odds) if odds and pd.notna(odds) else 1.91
            ev = float(cover_prob(edge_pts, MARGIN_COVER_STD)) * odds - 1.0
            return {"side": side, "line": float(line), "odds": odds,
                    "book": (best["book"] if best else (r.get("au_book") or "Sportsbet")),
                    "stake": bk.kelly_stake(ev, odds, bank_total), "ev": ev, "model": model,
                    "team": short(r["Home Team"]) if side == "home" else short(r["Away Team"])}
        pick_form = _form(pick["side"], pick["line"], pick["edge"], "model") if pick else None
        _gk = (r["Home Team"], r["Away Team"])

        def _tform(over: bool, line, edge_pts, model):
            shop = _au_shop_totals(books, over)
            best = shop[0] if shop else None
            odds = (best["price"] if best and best["price"] else (r.get("au_over") if over else r.get("au_under")))
            odds = float(odds) if odds and pd.notna(odds) else 1.91
            ev = float(cover_prob(edge_pts, TOTAL_COVER_STD)) * odds - 1.0
            return {"side": "over" if over else "under", "line": float(best["point"] if best else line),
                    "odds": odds, "book": (best["book"] if best else (r.get("au_book") or "Sportsbet")),
                    "stake": bk.kelly_stake(ev, odds, bank_total), "ev": ev, "model": model,
                    "team": ("OVER" if over else "UNDER")}
        tot_pick = tot_form = soft_tot = soft_tot_form = None
        tsig = total_signal(t_edge) if pd.notna(t_edge) else None
        if tsig:
            tot_pick = {"side": tsig, "line": float(r["au_total"]), "edge": abs(float(t_edge))}
            tot_form = _tform(tsig == "over", r["au_total"], abs(float(t_edge)), "model_total")
        tgap = (float(r["au_total"]) - float(r["sharp_total"])) if pd.notna(r.get("au_total")) and pd.notna(r.get("sharp_total")) else np.nan
        if pd.notna(tgap) and abs(tgap) >= TOTAL_LAG_MIN:
            soft_tot = {"side": "under" if tgap > 0 else "over", "line": float(r["au_total"]), "gap": abs(tgap)}
            soft_tot_form = _tform(tgap < 0, r["au_total"], abs(tgap), "soft_total")
        soft_form = (_form("home" if soft["team"] == short(r["Home Team"]) else "away",
                           soft["line"], soft["gap"], "soft") if soft else None)
        for _frm in (pick_form, soft_form, tot_form, soft_tot_form):
            if _frm:
                _frm["taken"] = (*_gk, _frm["side"]) in placed
        cards.append({
            "my_bets": my_bets.get(_gk, []),
            "tshop": _totals_shop(books),
            "pick_form": pick_form, "soft_form": soft_form, "tot_pick": tot_pick, "tot_form": tot_form,
            "soft_tot": soft_tot, "soft_tot_form": soft_tot_form, "sharp_total": r.get("sharp_total"),
            "wx": ("enclosed" if r.get("enclosed") in (True, 1) else
                   ("venue unknown" if r.get("enclosed") is None or (isinstance(r.get("enclosed"), float) and pd.isna(r.get("enclosed")))
                    else (f"wind {float(r['wind']):.0f} mph" + (f", gust {float(r['gust_mph']):.0f}" if pd.notna(r.get('gust_mph')) else "")
                          if pd.notna(r.get("wind")) else "outdoor, no forecast yet"))),
            "venue": r.get("venue") or "",
            "windy": bool(pd.notna(r.get("wind")) and float(r.get("wind") or 0) >= 15 and r.get("enclosed") is False),
            "gd": gd, "kick": ko.strftime("%a %d %b %H:%M") if pd.notna(ko) else "–",
            "home": r["Home Team"], "away": r["Away Team"], "h": short(r["Home Team"]), "a": short(r["Away Team"]),
            "au_book": r.get("au_book") or "AU", "au": _hcap(r["Home Team"], r["Away Team"], r.get("au_point")),
            "au_odds": f"{_f(r.get('au_home_odds'),2)} / {_f(r.get('au_away_odds'),2)}",
            "sharp_book": (lambda sb: ("sharp (median of %s)" % sb) if "," in sb else sb)(r.get("sharp_book") if isinstance(r.get("sharp_book"), str) and r.get("sharp_book") else "Pinnacle"),
            "sharp": _hcap(r["Home Team"], r["Away Team"], r.get("sharp_point")),
            "sharp_detail": " · ".join(f"{k.replace(' (US)','')} {_hcap(r['Home Team'], r['Away Team'], v[0])}"
                                       for k, v in sorted((r.get("sharp_detail") or {}).items()) if v[0] is not None),
            "gap": r.get("au_vs_sharp"), "sharp_call": _sharp_call(r),
            "model": _hcap(r["Home Team"], r["Away Team"], -r["pred_margin"] if pd.notna(r["pred_margin"]) else None),
            "model_raw": _hcap(r["Home Team"], r["Away Team"], -r["pred_raw"] if pd.notna(r["pred_raw"]) else None),
            "edge": r.get("edge"), "pick": pick,
            "cover": float(cover_prob(abs(r["edge"]), MARGIN_COVER_STD)) if pd.notna(r.get("edge")) else None,
            "p_home": r.get("home_win_prob"),
            "h2h": f"{_f(r.get('au_h2h_home'),2)} / {_f(r.get('au_h2h_away'),2)}",
            "elo": _signed(r.get("elo_diff"), 0), "form": _signed(r.get("form_diff")),
            "rest": _signed(r.get("rest_diff"), 0), "tz": _signed(r.get("tz_shift"), 0),
            "out_diff": r.get("player_out_diff"), "home_outs": r.get("home_outs") or [],
            "away_outs": r.get("away_outs") or [], "bk_home": r.get("backup_qb_home"),
            "bk_away": r.get("backup_qb_away"), "qb_coef": r.get("qb_coef"),
            "au_total": r.get("au_total"), "pred_total": r.get("pred_total"), "t_edge": t_edge,
            "n_au": r.get("n_au_books"), "n_books": r.get("n_books"),
            "shop": _au_shop(r.get("books"), pick["side"] == "home") if pick else
                    (_au_shop(r.get("books"), True) if _sharp_call(r) and _sharp_call(r)["team"] == short(r["Home Team"])
                     else _au_shop(r.get("books"), False) if _sharp_call(r) else []),
            "history": _line_history(gd, r["Home Team"], r["Away Team"]),
        })
    return cards, warn, placed


PAGE = r"""
<!doctype html><html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>NFL Line</title>
<style>
:root{--bg:#0f1418;--card:#171e25;--line:#2a343e;--fg:#e6edf3;--mut:#8b98a5;--ok:#3fb950;--warn:#d29922;--bad:#f85149;--acc:#58a6ff;--zap:#e3b341}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--fg);font:14px/1.45 -apple-system,Segoe UI,Roboto,sans-serif}
a{color:var(--acc)}h1{font-size:20px;margin:0}h2{font-size:15px;margin:22px 0 8px;color:var(--mut);text-transform:uppercase;letter-spacing:.06em}
.wrap{max-width:1180px;margin:0 auto;padding:16px}
.top{display:flex;flex-wrap:wrap;gap:8px 18px;align-items:baseline;border-bottom:1px solid var(--line);padding-bottom:10px}
.pill{font-size:12px;color:var(--mut)}.pill b{color:var(--fg)}.ok{color:var(--ok)}.warn{color:var(--warn)}.bad{color:var(--bad)}
.banner{background:#2b2111;border:1px solid var(--warn);padding:8px 12px;border-radius:6px;margin:12px 0}
.card{background:var(--card);border:1px solid var(--line);border-radius:8px;padding:12px 14px;margin:10px 0}
.card.sig{border-color:var(--ok)}.card.tsig{box-shadow:inset 3px 0 0 var(--acc)}.card.zap{border-color:var(--zap)}
.hdr{display:flex;flex-wrap:wrap;justify-content:space-between;gap:6px;align-items:baseline}
.teams{font-size:16px;font-weight:600}.kick{color:var(--mut);font-size:12px}
.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(150px,1fr));gap:8px 14px;margin-top:8px}
.k{font-size:11px;color:var(--mut);text-transform:uppercase;letter-spacing:.04em}.v{font-size:15px}
.bet{display:inline-block;background:#13361f;border:1px solid var(--ok);color:var(--ok);padding:3px 9px;border-radius:5px;font-weight:600}
.soft{display:inline-block;background:#3a2f0b;border:1px solid var(--zap);color:var(--zap);padding:3px 9px;border-radius:5px;font-weight:600}
.nobet{color:var(--mut)}
table{border-collapse:collapse;width:100%;font-size:13px}th,td{text-align:left;padding:5px 8px;border-bottom:1px solid var(--line)}th{color:var(--mut);font-weight:500;font-size:12px}
td.num,th.num{text-align:right}
.outs{font-size:12px;color:var(--mut);margin-top:6px}.outs b{color:var(--fg)}.qb{color:var(--bad);font-weight:600}
details{margin-top:6px}summary{cursor:pointer;color:var(--acc);font-size:12px}
form.inline{display:inline}input,select{background:#0d1117;color:var(--fg);border:1px solid var(--line);border-radius:4px;padding:5px 7px;font-size:13px}
button{background:#21313f;color:var(--fg);border:1px solid var(--line);border-radius:4px;padding:5px 10px;cursor:pointer;font-size:13px}
button.pri{background:#1f4d2e;border-color:var(--ok)}button.danger{background:#3a1d1d;border-color:var(--bad)}
.row{display:flex;flex-wrap:wrap;gap:8px;align-items:center;margin:6px 0}
.small{font-size:12px;color:var(--mut)}code{background:#0d1117;padding:1px 4px;border-radius:3px}
.spark{font-family:ui-monospace,monospace;font-size:12px;color:var(--mut)}
.taken{font-size:12px;color:#ffd866;background:#2e2410;padding:2px 8px;border-radius:6px;font-weight:600}
.quick{background:#0d1117;border:1px dashed var(--line);border-radius:6px;padding:4px 8px;margin-top:6px}
.bank{display:flex;flex-wrap:wrap;gap:8px 16px;align-items:center;background:#10181f;border:1px solid #244055;border-radius:8px;padding:8px 12px;margin:10px 0}
.bank b{font-size:16px}
</style></head><body><div class="wrap">
<div class="top">
 <h1>🏈 NFL Line <span class="small">AU book vs sharp vs model</span></h1>
 <span class="pill">history <b>{{st.games}}</b> games · last <b>{{st.last_result}}</b> · nflverse join <b>{{st.nflverse_match}}</b> ({{st.nflverse_age}})</span>
 <span class="pill">odds cache <b>{{st.odds_age}}</b> · {{st.credits_per_call}} credits/call{% if st.api %} · quota used <b>{{st.api.used}}</b> left <b class="{{'bad' if (st.api.remaining or 0) < 60 else 'ok'}}">{{st.api.remaining}}</b>{% endif %}</span>
 <span class="pill">injuries {% if st.injury %}<b class="{{'ok' if st.injury.ok else 'bad'}}">{{st.injury.outs}} outs</b> · {{st.injury.questionable}} Q · {{st.injury.ts}}{% else %}<b class="warn">not synced</b>{% endif %}</span>
 <span class="pill">ntfy <b class="{{'ok' if st.ntfy else 'warn'}}">{{'on' if st.ntfy else 'off'}}</b></span>
</div>
{% if warn %}<div class="banner">{{warn}}</div>{% endif %}
{% if not st.xlsx %}<div class="banner">Missing <code>data/nfl_results_and_odds.xlsx</code> — download from aussportsbetting.com (see README).</div>{% endif %}

<div class="row">
 <form class="inline" method="post" action="{{url_for('refresh_injuries')}}"><button>Sync injuries (ESPN, free)</button></form>
 <form class="inline" method="post" action="{{url_for('sync_results')}}"><button>Sync results (ESPN, free)</button></form>
 <form class="inline" method="post" action="{{url_for('sync_nflverse')}}"><button>Refresh nflverse</button></form>
 <form class="inline" method="post" action="{{url_for('refresh_odds')}}" onsubmit="return confirm('Fresh Odds API pull costs {{st.credits_per_call}} credits. Continue?')"><button>Fresh odds pull ({{st.credits_per_call}} credits)</button></form>
 <form class="inline" method="post" action="{{url_for('run_backtest')}}"><button>Run backtest</button></form>
</div>

<div class="bank">
 {% if bank %}<span>💰 Bankroll <b>{{'%.2f'|format(bank.total)}}</b></span>
  <span class="small">= unstaked {{'%.2f'|format(bank.unstaked)}} + pending {{'%.2f'|format(bank.pending)}} ({{bank.n_pending}} open) · since {{bank.anchor_date}}: {{'%+.2f'|format(bank.since_anchor)}}{% if bank.shared %} (shared with NRL · NFL {{'%+.2f'|format(bank.own_since_anchor)}}){% endif %} · peak {{'%.2f'|format(bank.peak)}} · max drawdown {{'%.2f'|format(bank.max_dd)}}</span>
  {% if bank.spark %}<span class="spark">{{bank.spark}}</span>{% endif %}
 {% else %}<span class="small">No bankroll set — enter your current unstaked balance to get Kelly stake suggestions.</span>{% endif %}
 <form class="inline" method="post" action="{{url_for('set_bankroll')}}">unstaked <input name="unstaked" type="number" step="0.01" style="width:90px" value="{{'%.2f'|format(bank.unstaked) if bank else ''}}"> <button>update</button></form>
 <form class="inline" method="post" action="{{url_for('set_kelly')}}">stake <select name="fraction" onchange="this.form.submit()">
  {% for v,l in [(1,'full Kelly'),(0.5,'½ Kelly'),(0.25,'¼ Kelly'),(0.125,'⅛ Kelly')] %}<option value="{{v}}" {{'selected' if (kelly-v)|abs < 0.01}}>{{l}}</option>{% endfor %}</select></form>
</div>

<h2>Board — next {{show_days}} days · handicap: fav ≥ {{fav}} / dog ≥ {{dog}} · totals: under ≥ {{tu}} / over ≥ {{to}} (wind-aware) · soft when AU trails Pinnacle by ≥ {{lag}} (totals {{tlag}})</h2>
{% if not cards %}<div class="card">No upcoming games to show.</div>{% endif %}
{% for c in cards %}
<div class="card {{'sig' if c.pick else ('zap' if c.sharp_call else '')}} {{'tsig' if (c.tot_pick or c.soft_tot) else ''}}">
 <div class="hdr">
  <div>{% if c.my_bets %}<span class="taken">✓ on: {{c.my_bets|join(', ')}}</span> {% endif %}<span class="teams">{{c.a}} @ {{c.h}}</span> <span class="kick">{{c.kick}} AU · {{c.n_au}} AU books / {{c.n_books}} total</span></div>
  <div>
   {% if c.pick %}<span class="bet">BET {{c.pick.team}} {{'%+.1f'|format(c.pick.line)}} @ {{c.au_book}} {{'%.2f'|format(c.pick.odds) if c.pick.odds else ''}} · edge {{'%.1f'|format(c.pick.edge)}} · cover {{'%.0f'|format(c.cover*100)}}%</span>
   {% else %}<span class="nobet">no model bet (edge {{'%+.1f'|format(c.edge) if c.edge==c.edge else '–'}})</span>{% endif %}
   {% if c.sharp_call %} <span class="soft">AU SOFT: {{c.sharp_call.team}} {{'%+.1f'|format(c.sharp_call.line)}} ({{'%+.1f'|format(c.sharp_call.gap)}} vs sharp)</span>{% endif %}
   {% if c.tot_pick %} <span class="bet">TOTAL: {{c.tot_pick.side|upper}} {{'%.1f'|format(c.tot_pick.line)}} · edge {{'%.1f'|format(c.tot_pick.edge)}}</span>{% endif %}
   {% if c.soft_tot %} <span class="soft">AU SOFT TOTAL: {{c.soft_tot.side|upper}} {{'%.1f'|format(c.soft_tot.line)}} ({{'%+.1f'|format(c.soft_tot.gap)}} vs sharp)</span>{% endif %}
  </div>
 </div>
 {% for f in [c.pick_form, c.soft_form, c.tot_form, c.soft_tot_form] if f %}
 {% if f.taken %}<div class="row quick"><span class="small">{{f.model|replace('_',' ')}} · {{f.team}}</span> <span class="taken">✓ already bet {{f.team}}</span></div>{% else %}
 <form class="row quick" method="post" action="{{url_for('quicklog')}}">
  <input type="hidden" name="gdate" value="{{c.gd}}"><input type="hidden" name="home" value="{{c.home}}"><input type="hidden" name="away" value="{{c.away}}">
  <input type="hidden" name="side" value="{{f.side}}"><input type="hidden" name="model" value="{{f.model}}">
  <span class="small">{{f.model|replace('_',' ')}} · {{f.team}}</span>
  <input name="line" type="number" step="0.5" value="{{'%.1f'|format(f.line)}}" style="width:70px" title="line">
  @<input name="odds" type="number" step="0.01" value="{{'%.2f'|format(f.odds)}}" style="width:70px" title="price">
  ×<input name="stake" type="number" step="1" min="1" value="{{f.stake or 1}}" style="width:60px" title="stake (Kelly suggestion)">
  <input name="book" value="{{f.book}}" style="width:110px" title="book">
  <button class="pri">bet</button>
  <span class="small">EV {{'%+.1f'|format(f.ev*100)}}%{% if f.stake %} · Kelly {{f.stake}}{% else %} · set a bankroll for Kelly{% endif %}</span>
 </form>{% endif %}
 {% endfor %}
 <div class="grid">
  <div><div class="k">{{c.au_book}} line</div><div class="v">{{c.au}} <span class="small">{{c.au_odds}}</span></div></div>
  <div><div class="k">{{c.sharp_book}} line</div><div class="v">{% if c.gap==c.gap %}{{c.sharp}} <span class="small">gap {{'%+.1f'|format(c.gap)}}</span><div class="small">{{c.sharp_detail}}</div>{% else %}<span class="small">no sharp line yet</span>{% endif %}</div></div>
  <div><div class="k">Model (blend {{blend}})</div><div class="v">{{c.model}} <span class="small">raw {{c.model_raw}}</span></div></div>
  <div><div class="k">P(home) · H2H</div><div class="v">{{'%.0f'|format(c.p_home*100)}}% <span class="small">{{c.h2h}}</span></div></div>
  <div><div class="k">Total AU / sharp / model</div><div class="v">{{'%.1f'|format(c.au_total) if c.au_total==c.au_total else '–'}} / {{'%.1f'|format(c.sharp_total) if c.sharp_total==c.sharp_total else '–'}} / {{'%.1f'|format(c.pred_total) if c.pred_total==c.pred_total else '–'}} <span class="small">{{'%+.1f'|format(c.t_edge) if c.t_edge==c.t_edge else ''}}</span></div></div>
  <div><div class="k">Elo · form · rest · tz</div><div class="v small">{{c.elo}} · {{c.form}} · {{c.rest}}d · {{c.tz}}h</div></div>
  <div><div class="k">Venue · weather</div><div class="v small">{{c.venue[:30]}} · <span class="{{'warn' if c.windy else ''}}">{{c.wx}}</span></div></div>
 </div>
 <div class="outs">
  Outs: <b>{{c.h}}</b>
  {% for o in c.home_outs %}{% if o.qb %}<span class="qb">{{o.player}} (QB starter → model {{'%+.1f'|format(-c.qb_coef)}})</span>{% else %}{{o.player}} {{o.position}} {{'%+.1f'|format(-o.points)}}{% endif %}{{', ' if not loop.last}}{% else %}none{% endfor %}
  &nbsp;|&nbsp; <b>{{c.a}}</b>
  {% for o in c.away_outs %}{% if o.qb %}<span class="qb">{{o.player}} (QB starter → model {{'%+.1f'|format(-c.qb_coef)}})</span>{% else %}{{o.player}} {{o.position}} {{'%+.1f'|format(-o.points)}}{% endif %}{{', ' if not loop.last}}{% else %}none{% endfor %}
  &nbsp;·&nbsp; net to home margin {{'%+.1f'|format(c.out_diff)}}
 </div>
 {% if c.shop %}<details><summary>AU shop for {{(c.pick.team if c.pick else c.sharp_call.team)}} ({{c.shop|length}} books)</summary>
  <table><tr><th>Book</th><th class="num">Line</th><th class="num">Price</th></tr>
  {% for s in c.shop %}<tr><td>{{s.book}}</td><td class="num">{{'%+.1f'|format(s.point)}}</td><td class="num">{{'%.2f'|format(s.price) if s.price else '–'}}</td></tr>{% endfor %}</table></details>{% endif %}
 {% if c.tshop %}<details><summary>Shop totals — best OVER {{'%.1f'|format(c.tshop.best_over.point)}} @ {{'%.2f'|format(c.tshop.best_over.over) if c.tshop.best_over.over else '–'}} {{c.tshop.best_over.book}} · best UNDER {{'%.1f'|format(c.tshop.best_under.point)}} @ {{'%.2f'|format(c.tshop.best_under.under) if c.tshop.best_under.under else '–'}} {{c.tshop.best_under.book}} ({{c.tshop.n}} AU books, {{'%.1f'|format(c.tshop.lo)}}–{{'%.1f'|format(c.tshop.hi)}})</summary>
  <table><tr><th>Book</th><th class="num">Total</th><th class="num">Over</th><th class="num">Under</th></tr>
  {% for s in c.tshop.rows %}<tr{% if s.sharp %} class="small"{% endif %}><td>{{s.book}}{% if s.sharp %} (sharp ref){% endif %}</td><td class="num">{{'%.1f'|format(s.point)}}</td>
   <td class="num {{'ok' if s.best_over}}">{{'%.2f'|format(s.over) if s.over else '–'}}{{' ★' if s.best_over}}</td>
   <td class="num {{'ok' if s.best_under}}">{{'%.2f'|format(s.under) if s.under else '–'}}{{' ★' if s.best_under}}</td></tr>{% endfor %}</table>
  <div class="row">{% for sd in ['over','under'] %}{% set b = c.tshop['best_' ~ sd] %}{% if (c.home, c.away, sd) in placed %}<span class="taken">✓ already bet {{sd|upper}}</span>{% else %}
   <form class="inline" method="post" action="{{url_for('quicklog')}}">
    <input type="hidden" name="gdate" value="{{c.gd}}"><input type="hidden" name="home" value="{{c.home}}"><input type="hidden" name="away" value="{{c.away}}">
    <input type="hidden" name="side" value="{{sd}}"><input type="hidden" name="model" value="shop_total"><input type="hidden" name="book" value="{{b.book}}">
    {{sd|upper}} <input name="line" type="number" step="0.5" value="{{'%.1f'|format(b.point)}}" style="width:65px">
    @<input name="odds" type="number" step="0.01" value="{{'%.2f'|format(b[sd]) if b[sd] else '1.90'}}" style="width:65px">
    ×<input name="stake" type="number" step="1" min="1" value="10" style="width:55px"> <button>bet {{sd}} @ {{b.book}}</button></form>{% endif %}
  {% endfor %}</div></details>{% endif %}
 {% if c.history %}<details><summary>Line history ({{c.history|length}} captures)</summary>
  <table><tr><th>Phase</th><th>When</th><th class="num">AU home pt</th><th class="num">Sharp home pt</th></tr>
  {% for hst in c.history %}<tr><td>{{hst.phase}}</td><td>{{hst.ts}}</td><td class="num">{{'%+.1f'|format(hst.au) if hst.au is not none else '–'}}</td><td class="num">{{'%+.1f'|format(hst.sharp) if hst.sharp is not none else '–'}}</td></tr>{% endfor %}</table></details>{% endif %}
</div>
{% endfor %}

<h2>My bets</h2>
<div class="card">
 <form method="post" action="{{url_for('add_bet')}}" class="row">
  <input name="gdate" type="date" required><select name="home" required><option value="">home…</option>{% for t in teams %}<option>{{t}}</option>{% endfor %}</select>
  <select name="away" required><option value="">away…</option>{% for t in teams %}<option>{{t}}</option>{% endfor %}</select>
  <select name="side"><option value="home">home</option><option value="away">away</option><option value="over">over</option><option value="under">under</option></select>
  <input name="line" type="number" step="0.5" placeholder="line (+/-) or total" style="width:130px" required>
  <input name="odds" type="number" step="0.01" placeholder="odds" style="width:80px" value="1.91">
  <input name="stake" type="number" step="1" placeholder="stake" style="width:80px" required>
  <input name="book" placeholder="book" style="width:110px" value="{{au_book}}">
  <button class="pri">Log</button>
 </form>
 {% if bets %}<table><tr><th>Game</th><th>Bet</th><th>Src</th><th class="num">Odds</th><th class="num">Stake</th><th>Book</th><th>Score</th><th>Result</th><th class="num">P/L</th><th class="num">CLV</th><th></th></tr>
 {% for b in bets %}<tr><td>{{b.gdate}} {{b.away_s}} @ {{b.home_s}}</td><td>{{b.team_s}} {{('%.1f' if b.is_total else '%+.1f')|format(b.line)}}</td><td class="small">{{b.model or 'manual'}}</td><td class="num">{{'%.2f'|format(b.odds)}}</td><td class="num">{{'%.0f'|format(b.stake)}}</td><td>{{b.book}}</td>
  <td>{{b.score}}</td><td class="{{'ok' if b.result=='W' else ('bad' if b.result=='L' else '')}}">{{b.result or 'pending'}}</td><td class="num">{{'%+.0f'|format(b.profit) if b.profit is not none else ''}}</td><td class="num">{{b.clv}}</td>
  <td><form class="inline" method="post" action="{{url_for('remove_bet')}}"><input type="hidden" name="rowid" value="{{b.rowid}}"><button class="danger">×</button></form></td></tr>{% endfor %}</table>
 <table style="margin-top:10px"><tr><th>Performance by source</th><th class="num">bets</th><th class="num">settled</th><th class="num">W-L-P</th><th class="num">P/L</th><th class="num">ROI</th><th class="num">avg CLV</th><th class="num">beat close</th></tr>
 {% for t in tally %}<tr><td>{{t.label}}</td><td class="num">{{t.n}}</td><td class="num">{{t.settled}}</td><td class="num">{{t.w}}-{{t.l}}-{{t.p}}</td><td class="num {{'ok' if t.pl>0 else ('bad' if t.pl<0 else '')}}">{{'%+.2f'|format(t.pl)}}</td><td class="num">{{'%+.1f'|format(t.roi*100)}}%</td><td class="num">{{t.clv}}</td><td class="num">{{t.beat}}</td></tr>{% endfor %}</table>
 {% else %}<div class="small">No bets logged.</div>{% endif %}
</div>

<h2>Signal scorecard — every flagged signal, taken or not</h2>
<div class="card">
 {% if scorecards %}<table><tr><th>Signal</th><th class="num">n</th><th class="num">avg CLV (pts)</th><th class="num">beat close</th><th class="num">settled</th><th class="num">cover</th><th class="num">ROI @1.91</th></tr>
 {% for sc in scorecards %}<tr><td>{{sc.label}}</td><td class="num">{{sc.n}}</td><td class="num {{'ok' if sc.pos else 'bad'}}">{{sc.avg_clv}}</td><td class="num">{{sc.beat}}</td><td class="num">{{sc.settled or '–'}}</td><td class="num">{{sc.cover or '–'}}</td><td class="num">{{sc.roi or '–'}}</td></tr>{% endfor %}</table>
 <div class="small" style="margin-top:6px">CLV = the pick's AU line at the first flag vs the last AU capture (positive = the line moved toward the pick). Cover/ROI once results land. This is the unbiased read on both the model and the "AU is soft" thesis.</div>
 {% else %}<div class="small">Needs at least two captures per game (first poll vs close). Accumulates from the first cron poll.</div>{% endif %}
</div>

<h2>Backtest &amp; market structure</h2>
<div class="card">
{% if bt %}
 <div class="small">Walk-forward {{bt.seasons[0]}}–{{bt.seasons[-1]}} ({{bt.games}} games), generated {{bt.generated}}</div>
 <div class="grid">
  {% for k,v in bt.mae.items() %}<div><div class="k">MAE {{k}}</div><div class="v">{{'%.2f'|format(v)}}</div></div>{% endfor %}
 </div>
 <table style="margin-top:8px"><tr><th>edge ≥</th><th class="num">bets</th><th class="num">cover vs open</th><th class="num">ROI vs open</th><th class="num">cover vs close</th><th class="num">ROI vs close</th></tr>
 {% for t,r in bt.edge_sweep.items() %}<tr><td>{{t}}</td><td class="num">{{r.vs_open.bets}}</td><td class="num">{{'%.1f'|format((r.vs_open.cover or 0)*100)}}%</td><td class="num">{{'%+.1f'|format((r.vs_open.roi or 0)*100)}}%</td><td class="num">{{'%.1f'|format((r.vs_close.cover or 0)*100)}}%</td><td class="num">{{'%+.1f'|format((r.vs_close.roi or 0)*100)}}%</td></tr>{% endfor %}</table>
 <div style="margin-top:10px"><b>Market structure — workbook lines</b> (Pinnacle 2014-18, bet365 2018-25, Betr Sep 2025+; NOT AU books historically — the AU-vs-sharp test is the forward capture below each game) · breakeven at -110 = 52.4%<br>
  <span class="small">MAE open {{'%.2f'|format(bt.market.mae_open)}} vs close {{'%.2f'|format(bt.market.mae_close)}} ·
  follow-the-move at the open (|move| ≥ 1): {{bt.market.follow_move_bets}} bets, cover <b>{{'%.1f'|format(bt.market.follow_move_cover*100)}}%</b>
  {% if bt.market.us_close_vs_au_open_cover is defined %}· back the US-close side at the open (|gap| ≥ 1): {{bt.market.us_close_vs_au_open_games}} games, cover <b>{{'%.1f'|format(bt.market.us_close_vs_au_open_cover*100)}}%</b> · MAE US close {{'%.2f'|format(bt.market.mae_us_close)}} vs workbook close {{'%.2f'|format(bt.market.mae_au_close_same)}}{% endif %}</span></div>
{% else %}<div class="small">No backtest yet — click "Run backtest" (~20s).</div>{% endif %}
</div>

{% if bt and bt.rules %}<h2>Live signal rules — walk-forward, chosen on 2012-18, checked on 2019-25</h2>
<div class="card"><table><tr><th>Rule</th><th class="num">train n</th><th class="num">train cover</th><th class="num">train ROI</th><th class="num">validate n</th><th class="num">validate cover</th><th class="num">validate ROI</th></tr>
{% for r in bt.rules %}<tr><td>{{r.name}}</td><td class="num">{{r.train_n}}</td><td class="num">{{'%.1f'|format(r.train_cover*100)}}%</td><td class="num">{{'%+.1f'|format(r.train_roi*100)}}%</td><td class="num">{{r.val_n}}</td><td class="num">{{'%.1f'|format(r.val_cover*100)}}%</td><td class="num {{'ok' if r.val_roi>0 else 'bad'}}">{{'%+.1f'|format(r.val_roi*100)}}%</td></tr>{% endfor %}</table>
<div class="small" style="margin-top:6px">Settled at 1.91 vs the workbook OPENING line (Pinnacle/bet365/Betr). Edges shrink by kickoff — bet early, judge live by CLV.</div></div>{% endif %}

<h2>Model weights (line-free linear, latest fit)</h2>
<div class="card small">{% for f,c in coeffs %}<span style="display:inline-block;margin:2px 12px 2px 0"><b>{{f}}</b> {{'%+.3f'|format(c)}}</span>{% else %}not fitted yet{% endfor %}
 <div style="margin-top:6px">backup_qb_diff is the learned cost of a backup QB starting (pts of margin); positive = the side with the starter gains.</div></div>
<h2>Player outs (manual + ESPN)</h2>
<div class="card">
 <form method="post" action="{{url_for('add_out')}}" class="row">
  <select name="team" required><option value="">team…</option>{% for t in teams %}<option>{{t}}</option>{% endfor %}</select>
  <input name="player" placeholder="player" required>
  <select name="position">{% for p in positions %}<option>{{p}}</option>{% endfor %}</select>
  <select name="kind"><option value="out">out</option><option value="in">returning (in)</option></select>
  <input name="weeks" type="number" step="1" min="1" value="1" style="width:70px" title="weeks"> wk
  <input name="points" type="number" step="0.1" placeholder="pts (blank = position default)" style="width:200px">
  <button class="pri">Add</button>
  <span class="small">QB = the starter (uses the learned coefficient); others add their points to the margin.</span>
 </form>
 {% if outs %}<table><tr><th>Team</th><th>Player</th><th>Pos</th><th class="num">Pts</th><th>Kind</th><th>Since</th><th class="num">Wks</th><th>Source</th><th>Note</th><th></th></tr>
 {% for o in outs %}<tr><td>{{o.team}}</td><td>{{o.player}}</td><td>{{o.position}}</td><td class="num">{{'%.2f'|format(o.points or 0)}}</td><td>{{o.kind}}</td><td>{{o.entered}}</td><td class="num">{{'%.0f'|format(o.weeks or 1)}}</td><td>{{o.source}}</td><td class="small">{{(o.note or '')[:70]}}</td>
  <td><form class="inline" method="post" action="{{url_for('remove_out')}}"><input type="hidden" name="rowid" value="{{o.rowid}}"><button class="danger">×</button></form></td></tr>{% endfor %}</table>
 {% else %}<div class="small">No outs yet — click "Sync injuries".</div>{% endif %}
 {% if watch %}<details><summary>Questionable / watch-list ({{watch|length}})</summary><table>
  {% for w in watch %}<tr><td>{{w.team}}</td><td>{{w.player}}</td><td>{{w.position}}</td><td>{{w.status}}</td><td class="small">{{w.detail}}</td></tr>{% endfor %}</table></details>{% endif %}
</div>

<div class="small" style="margin:18px 0">Analytics, not advice. AU regions: {{regions}} · markets: {{markets}}. Gambling Help Online 1800 858 858.</div>
</div></body></html>
"""

POSITIONS = ["QB", "RB", "WR", "TE", "T", "G", "C", "DE", "DT", "LB", "CB", "S", "K", "P"]


@app.route("/")
def index():
    cards, warn, placed = get_board()
    outs_df = db.read_outs()
    outs = []
    for _, o in outs_df.sort_values(["team", "points"], ascending=[True, False]).iterrows():
        try:
            ent = pd.to_datetime(o["entered"]); wk = float(o["weeks"] or 1)
            if ent + pd.Timedelta(days=7 * wk) < _now():
                continue                          # expired window
        except Exception:  # noqa: BLE001
            pass
        outs.append(o.to_dict())
    bets_df = settle_bets(db.read_bets())
    bets = []
    for _, b in bets_df.sort_values("gdate", ascending=False).iterrows():
        d = b.to_dict()
        d["home_s"], d["away_s"] = short(b["home"]), short(b["away"])
        d["team_s"] = ({"home": short(b["home"]), "away": short(b["away"]), "over": "OVER", "under": "UNDER"}
                       .get(b["side"], b["side"]))
        d["is_total"] = b["side"] in ("over", "under")
        d["score"] = (f"{int(b['home_score'])}-{int(b['away_score'])}" if pd.notna(b.get("home_score")) else "")
        d["profit"] = b["profit"] if pd.notna(b.get("profit")) else None
        if pd.notna(b.get("close_line")):
            clv = float(b["line"]) - float(b["close_line"])
            if b["side"] == "over":
                clv = -clv
            d["clv"] = f"{clv:+.1f}"
        else:
            d["clv"] = ""
        bets.append(d)
    bank = None
    try:
        bank = bk.current()
        if bank:
            h = bk.history()
            tot = pd.to_numeric(h["total"], errors="coerce").dropna().tolist() if len(h) else []
            tot = tot + [bank["total"]]
            peak = max(tot); run_max = np.maximum.accumulate(tot)
            bank["peak"] = peak; bank["max_dd"] = float(np.max(run_max - np.array(tot)))
            bars = "▁▂▃▄▅▆▇█"
            lo, hi = min(tot), max(tot)
            bank["spark"] = "".join(bars[int((v - lo) / (hi - lo) * 7)] if hi > lo else bars[3] for v in tot[-40:])
    except Exception:  # noqa: BLE001
        bank = None
    try:
        scorecards = signal_scorecards()
    except Exception:  # noqa: BLE001
        scorecards = []
    bt = None
    try:
        p = DATA_DIR / "backtest_summary.json"
        bt = json.loads(p.read_text()) if p.exists() else None
    except Exception:  # noqa: BLE001
        bt = None
    return render_template_string(
        PAGE, st=_status(), warn=warn, cards=cards, show_days=SHOW_DAYS, signal=SIGNAL_EDGE,
        lag=SHARP_LAG_MIN, blend=MARKET_BLEND, tsignal=TOTAL_SIGNAL, tlag=TOTAL_LAG_MIN,
        fav=FAV_EDGE, dog=DOG_EDGE, tu=TOTAL_UNDER_EDGE, to=TOTAL_OVER_EDGE, teams=sorted(TEAMS), positions=POSITIONS, outs=outs,
        watch=db.get_json("injury_watch", []), bets=bets, tally=tally(bets_df), bt=bt, coeffs=_coeffs(),
        bank=bank, kelly=bk.kelly_fraction(), scorecards=scorecards, placed=placed,
        au_book=(cards[0]["au_book"] if cards else "Sportsbet"), regions=ODDS_REGIONS, markets=ODDS_MARKETS)


@app.route("/api/board")
def api_board():
    cards, warn, _ = get_board()
    return jsonify({"warn": warn, "games": [{k: v for k, v in c.items() if k not in ("history",)} for c in cards]})


@app.route("/add_out", methods=["POST"])
def add_out():
    team = canon_team(request.form.get("team"))
    pos = (request.form.get("position") or "").upper()
    pts = request.form.get("points")
    points = float(pts) if pts else POSITION_POINTS.get(pos, 0.5)
    db.add_out(_now().date().isoformat(), float(request.form.get("weeks") or 1), team,
               request.form.get("player", "").strip(), pos, points, request.form.get("kind", "out"), "manual", "")
    return redirect(url_for("index"))


@app.route("/remove_out", methods=["POST"])
def remove_out():
    db.remove_out(int(request.form["rowid"]))
    return redirect(url_for("index"))


@app.route("/refresh_injuries", methods=["POST"])
def refresh_injuries():
    try:
        from src.injuries import sync
        sync()
    except Exception as ex:  # noqa: BLE001
        db.set_json("injury_status", {"ok": False, "ts": str(_now())[:16], "outs": 0, "questionable": 0, "err": str(ex)})
    return redirect(url_for("index"))


@app.route("/sync_results", methods=["POST"])
def sync_results():
    try:
        from src.poll_results import fetch_scores
        db.capture_results(fetch_scores(7))
    except Exception:  # noqa: BLE001
        pass
    return redirect(url_for("index"))


@app.route("/sync_nflverse", methods=["POST"])
def sync_nflverse():
    try:
        from src.nflverse_sync import refresh
        refresh()
    except Exception:  # noqa: BLE001
        pass
    return redirect(url_for("index"))


@app.route("/refresh_odds", methods=["POST"])
def refresh_odds():
    try:
        events = fetch_odds(use_cache=False)
        db.snapshot_odds(events, phase="page")
        db.snapshot_book_lines(events, phase="page")
    except Exception:  # noqa: BLE001
        pass
    return redirect(url_for("index"))


@app.route("/backtest", methods=["POST"])
def run_backtest():
    try:
        from src.backtest import run
        run(verbose=False)
    except Exception as ex:  # noqa: BLE001
        (DATA_DIR / "backtest_error.txt").write_text(str(ex))
    return redirect(url_for("index"))


@app.route("/add_bet", methods=["POST"])
def add_bet():
    f = request.form
    side = f.get("side", "home")
    home, away = canon_team(f.get("home")), canon_team(f.get("away"))
    db.add_bet({"logged": str(_now())[:16], "gdate": f.get("gdate"), "home": home, "away": away, "side": side,
                "team": {"home": home, "away": away}.get(side, side.upper()), "line": float(f.get("line") or 0),
                "odds": float(f.get("odds") or 1.909), "stake": float(f.get("stake") or 0),
                "book": f.get("book") or "", "model": "manual"})
    return redirect(url_for("index"))


@app.route("/quicklog", methods=["POST"])
def quicklog():
    """One-click: log a flagged pick (model or AU-soft) at the shown price/stake."""
    f = request.form
    side = f.get("side", "home")
    home, away = canon_team(f.get("home")), canon_team(f.get("away"))
    try:
        line, odds, stake = float(f.get("line")), float(f.get("odds") or 1.909), float(f.get("stake") or 1)
    except (TypeError, ValueError):
        return redirect(url_for("index"))
    db.add_bet({"logged": str(_now())[:16], "gdate": f.get("gdate"), "home": home, "away": away, "side": side,
                "team": {"home": home, "away": away}.get(side, side.upper()), "line": line, "odds": odds,
                "stake": stake, "book": f.get("book") or "", "model": f.get("model") or "manual"})
    return redirect(url_for("index"))


@app.route("/set_bankroll", methods=["POST"])
def set_bankroll():
    try:
        bk.set_anchor(float(request.form.get("unstaked")))
    except (TypeError, ValueError):
        pass
    return redirect(url_for("index"))


@app.route("/set_kelly", methods=["POST"])
def set_kelly():
    try:
        v = float(request.form.get("fraction"))
        if 0 < v <= 1:
            db.set_kv("kelly_fraction", v)
    except (TypeError, ValueError):
        pass
    return redirect(url_for("index"))


@app.route("/remove_bet", methods=["POST"])
def remove_bet():
    db.remove_bet(int(request.form["rowid"]))
    return redirect(url_for("index"))


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000, debug=True)
