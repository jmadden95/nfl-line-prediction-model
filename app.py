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
                    SHARP_LAG_MIN, SIGNAL_EDGE)
from src import db
from src.live import cache_age_seconds, fetch_odds, mock_events, predict_upcoming
from src.teams import TEAMS, canon_team, short
from src.win_prob import cover_prob

app = Flask(__name__)
SHOW_DAYS = 8
MARGIN_COVER_STD = 20.0      # betting sigma for edge -> cover prob (wider than raw 14.6; NRL used 22 on 18.4)

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
    if e >= SIGNAL_EDGE:
        return {"team": short(r["Home Team"]), "line": ap, "side": "home", "edge": float(e),
                "odds": r.get("au_home_odds")}
    if e <= -SIGNAL_EDGE:
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


def _line_history(gd, home, away):
    snaps = db.odds_snapshots_df()
    if not len(snaps):
        return []
    g = snaps[(snaps["game_date"] == gd) & (snaps["home"] == home) & (snaps["away"] == away)]
    return [{"phase": s["phase"], "ts": str(s["captured_ts"])[5:16].replace("T", " "),
             "au": s["au_point"], "sharp": s["sharp_point"]} for _, s in g.sort_values("captured_ts").iterrows()]


def _settle_bets(bets: pd.DataFrame) -> pd.DataFrame:
    """Fill result/profit from known scores; close_line from the kickoff snapshot."""
    if not len(bets):
        return bets
    from src.load_data import load_raw
    try:
        m = load_raw()
    except Exception:  # noqa: BLE001
        return bets
    m = m.dropna(subset=["Home Score", "Away Score"])
    key = {(str(d.date()), h, a): (hs, as_) for d, h, a, hs, as_ in
           zip(m["Date"], m["Home Team"], m["Away Team"], m["Home Score"], m["Away Score"])}
    snaps = db.odds_snapshots_df()
    for i, b in bets.iterrows():
        if b.get("result") in ("W", "L", "P"):
            continue
        sc = key.get((str(b["gdate"])[:10], b["home"], b["away"]))
        if sc is None:
            continue
        hs, as_ = sc
        margin = hs - as_ if b["side"] == "home" else as_ - hs
        res = margin + float(b["line"])
        result = "W" if res > 0 else ("L" if res < 0 else "P")
        odds = float(b["odds"] or 1.909)
        stake = float(b["stake"] or 0)
        profit = stake * (odds - 1) if result == "W" else (-stake if result == "L" else 0.0)
        close = None
        if len(snaps):
            k = snaps[(snaps["game_date"] == str(b["gdate"])[:10]) & (snaps["home"] == b["home"])
                      & (snaps["away"] == b["away"])].sort_values("captured_ts")
            if len(k):
                cp = k.iloc[-1]["au_point"]
                close = cp if b["side"] == "home" else (-cp if cp is not None else None)
        db.update_bet(int(b["rowid"]), home_score=hs, away_score=as_, result=result, profit=profit,
                      close_line=close)
        bets.loc[i, ["home_score", "away_score", "result", "profit", "close_line"]] = [hs, as_, result, profit, close]
    return bets


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
        return [], warn
    up = predict_upcoming(events)
    if not len(up):
        return [], warn or "No upcoming games in the odds feed."
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
    cards = []
    for _, r in up.iterrows():
        ko = pd.to_datetime(r["Kickoff"], errors="coerce")
        if ko is not None and not pd.isna(ko) and ko > horizon:
            continue
        pick = _pick(r)
        gd = str(pd.Timestamp(r["Date"]).date())
        t_edge = (r["pred_total"] - r["au_total"]) if pd.notna(r.get("au_total")) and pd.notna(r.get("pred_total")) else np.nan
        cards.append({
            "gd": gd, "kick": ko.strftime("%a %d %b %H:%M") if pd.notna(ko) else "–",
            "home": r["Home Team"], "away": r["Away Team"], "h": short(r["Home Team"]), "a": short(r["Away Team"]),
            "au_book": r.get("au_book") or "AU", "au": _hcap(r["Home Team"], r["Away Team"], r.get("au_point")),
            "au_odds": f"{_f(r.get('au_home_odds'),2)} / {_f(r.get('au_away_odds'),2)}",
            "sharp_book": r.get("sharp_book") or "Pinnacle",
            "sharp": _hcap(r["Home Team"], r["Away Team"], r.get("sharp_point")),
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
    return cards, warn


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
.card.sig{border-color:var(--ok)}.card.zap{border-color:var(--zap)}
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

<h2>Board — next {{show_days}} days · bet when |edge| ≥ {{signal}} · soft when AU trails sharp by ≥ {{lag}}</h2>
{% if not cards %}<div class="card">No upcoming games to show.</div>{% endif %}
{% for c in cards %}
<div class="card {{'sig' if c.pick else ('zap' if c.sharp_call else '')}}">
 <div class="hdr">
  <div><span class="teams">{{c.a}} @ {{c.h}}</span> <span class="kick">{{c.kick}} AU · {{c.n_au}} AU books / {{c.n_books}} total</span></div>
  <div>
   {% if c.pick %}<span class="bet">BET {{c.pick.team}} {{'%+.1f'|format(c.pick.line)}} @ {{c.au_book}} {{'%.2f'|format(c.pick.odds) if c.pick.odds else ''}} · edge {{'%.1f'|format(c.pick.edge)}} · cover {{'%.0f'|format(c.cover*100)}}%</span>
   {% else %}<span class="nobet">no model bet (edge {{'%+.1f'|format(c.edge) if c.edge==c.edge else '–'}})</span>{% endif %}
   {% if c.sharp_call %} <span class="soft">AU SOFT: {{c.sharp_call.team}} {{'%+.1f'|format(c.sharp_call.line)}} ({{'%+.1f'|format(c.sharp_call.gap)}} vs sharp)</span>{% endif %}
  </div>
 </div>
 <div class="grid">
  <div><div class="k">{{c.au_book}} line</div><div class="v">{{c.au}} <span class="small">{{c.au_odds}}</span></div></div>
  <div><div class="k">{{c.sharp_book}} line</div><div class="v">{% if c.gap==c.gap %}{{c.sharp}} <span class="small">gap {{'%+.1f'|format(c.gap)}}</span>{% else %}<span class="small">no sharp line yet</span>{% endif %}</div></div>
  <div><div class="k">Model (blend {{blend}})</div><div class="v">{{c.model}} <span class="small">raw {{c.model_raw}}</span></div></div>
  <div><div class="k">P(home) · H2H</div><div class="v">{{'%.0f'|format(c.p_home*100)}}% <span class="small">{{c.h2h}}</span></div></div>
  <div><div class="k">Total AU / model</div><div class="v">{{'%.1f'|format(c.au_total) if c.au_total==c.au_total else '–'}} / {{'%.1f'|format(c.pred_total) if c.pred_total==c.pred_total else '–'}} <span class="small">{{'%+.1f'|format(c.t_edge) if c.t_edge==c.t_edge else ''}}</span></div></div>
  <div><div class="k">Elo · form · rest · tz</div><div class="v small">{{c.elo}} · {{c.form}} · {{c.rest}}d · {{c.tz}}h</div></div>
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
 {% if c.history %}<details><summary>Line history ({{c.history|length}} captures)</summary>
  <table><tr><th>Phase</th><th>When</th><th class="num">AU home pt</th><th class="num">Sharp home pt</th></tr>
  {% for hst in c.history %}<tr><td>{{hst.phase}}</td><td>{{hst.ts}}</td><td class="num">{{'%+.1f'|format(hst.au) if hst.au is not none else '–'}}</td><td class="num">{{'%+.1f'|format(hst.sharp) if hst.sharp is not none else '–'}}</td></tr>{% endfor %}</table></details>{% endif %}
</div>
{% endfor %}

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

<h2>My bets</h2>
<div class="card">
 <form method="post" action="{{url_for('add_bet')}}" class="row">
  <input name="gdate" type="date" required><select name="home" required><option value="">home…</option>{% for t in teams %}<option>{{t}}</option>{% endfor %}</select>
  <select name="away" required><option value="">away…</option>{% for t in teams %}<option>{{t}}</option>{% endfor %}</select>
  <select name="side"><option value="home">home</option><option value="away">away</option></select>
  <input name="line" type="number" step="0.5" placeholder="line (+/-)" style="width:100px" required>
  <input name="odds" type="number" step="0.01" placeholder="odds" style="width:80px" value="1.91">
  <input name="stake" type="number" step="1" placeholder="stake" style="width:80px" required>
  <input name="book" placeholder="book" style="width:110px" value="{{au_book}}">
  <button class="pri">Log</button>
 </form>
 {% if bets %}<table><tr><th>Game</th><th>Bet</th><th class="num">Odds</th><th class="num">Stake</th><th>Book</th><th>Score</th><th>Result</th><th class="num">P/L</th><th class="num">CLV</th><th></th></tr>
 {% for b in bets %}<tr><td>{{b.gdate}} {{b.away_s}} @ {{b.home_s}}</td><td>{{b.team_s}} {{'%+.1f'|format(b.line)}}</td><td class="num">{{'%.2f'|format(b.odds)}}</td><td class="num">{{'%.0f'|format(b.stake)}}</td><td>{{b.book}}</td>
  <td>{{b.score}}</td><td class="{{'ok' if b.result=='W' else ('bad' if b.result=='L' else '')}}">{{b.result or 'pending'}}</td><td class="num">{{'%+.0f'|format(b.profit) if b.profit is not none else ''}}</td><td class="num">{{b.clv}}</td>
  <td><form class="inline" method="post" action="{{url_for('remove_bet')}}"><input type="hidden" name="rowid" value="{{b.rowid}}"><button class="danger">×</button></form></td></tr>{% endfor %}</table>
 <div class="small" style="margin-top:6px">Settled {{tally.n}} · W-L-P {{tally.w}}-{{tally.l}}-{{tally.p}} · P/L {{'%+.0f'|format(tally.pl)}} · ROI {{'%.1f'|format(tally.roi*100)}}% · avg CLV {{tally.clv}}</div>
 {% else %}<div class="small">No bets logged.</div>{% endif %}
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

<h2>Model weights (line-free linear, latest fit)</h2>
<div class="card small">{% for f,c in coeffs %}<span style="display:inline-block;margin:2px 12px 2px 0"><b>{{f}}</b> {{'%+.3f'|format(c)}}</span>{% else %}not fitted yet{% endfor %}
 <div style="margin-top:6px">backup_qb_diff is the learned cost of a backup QB starting (pts of margin); positive = the side with the starter gains.</div></div>
<div class="small" style="margin:18px 0">Analytics, not advice. AU regions: {{regions}} · markets: {{markets}}. Gambling Help Online 1800 858 858.</div>
</div></body></html>
"""

POSITIONS = ["QB", "RB", "WR", "TE", "T", "G", "C", "DE", "DT", "LB", "CB", "S", "K", "P"]


@app.route("/")
def index():
    cards, warn = get_board()
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
    bets_df = _settle_bets(db.read_bets())
    bets, tally = [], {"n": 0, "w": 0, "l": 0, "p": 0, "pl": 0.0, "roi": 0.0, "clv": "–"}
    clvs, staked = [], 0.0
    for _, b in bets_df.sort_values("gdate", ascending=False).iterrows():
        d = b.to_dict()
        d["home_s"], d["away_s"] = short(b["home"]), short(b["away"])
        d["team_s"] = short(b["home"]) if b["side"] == "home" else short(b["away"])
        d["score"] = (f"{int(b['home_score'])}-{int(b['away_score'])}" if pd.notna(b.get("home_score")) else "")
        d["profit"] = b["profit"] if pd.notna(b.get("profit")) else None
        if pd.notna(b.get("close_line")):
            clv = float(b["line"]) - float(b["close_line"])   # got more points than the close = +
            clvs.append(clv); d["clv"] = f"{clv:+.1f}"
        else:
            d["clv"] = ""
        if b.get("result") in ("W", "L", "P"):
            tally["n"] += 1; tally[b["result"].lower()] += 1
            tally["pl"] += float(b["profit"] or 0); staked += float(b["stake"] or 0)
        bets.append(d)
    tally["roi"] = tally["pl"] / staked if staked else 0.0
    tally["clv"] = f"{np.mean(clvs):+.2f}" if clvs else "–"
    bt = None
    try:
        p = DATA_DIR / "backtest_summary.json"
        bt = json.loads(p.read_text()) if p.exists() else None
    except Exception:  # noqa: BLE001
        bt = None
    return render_template_string(
        PAGE, st=_status(), warn=warn, cards=cards, show_days=SHOW_DAYS, signal=SIGNAL_EDGE,
        lag=SHARP_LAG_MIN, blend=MARKET_BLEND, teams=sorted(TEAMS), positions=POSITIONS, outs=outs,
        watch=db.get_json("injury_watch", []), bets=bets, tally=tally, bt=bt, coeffs=_coeffs(),
        au_book=(cards[0]["au_book"] if cards else "Sportsbet"), regions=ODDS_REGIONS, markets=ODDS_MARKETS)


@app.route("/api/board")
def api_board():
    cards, warn = get_board()
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
                "team": home if side == "home" else away, "line": float(f.get("line") or 0),
                "odds": float(f.get("odds") or 1.909), "stake": float(f.get("stake") or 0),
                "book": f.get("book") or ""})
    return redirect(url_for("index"))


@app.route("/remove_bet", methods=["POST"])
def remove_bet():
    db.remove_bet(int(request.form["rowid"]))
    return redirect(url_for("index"))


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000, debug=True)
