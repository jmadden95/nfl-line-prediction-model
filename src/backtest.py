"""Walk-forward backtest + the market-structure test.

For each season from 2012 on: train the LINE-FREE linear model on all prior
seasons (from TRAIN_FROM_SEASON), predict that season, blend toward the AU
opening line, and simulate betting the handicap at |edge| >= threshold. Bets
settle against BOTH the opening line (what you can actually get early) and the
closing line (the benchmark: if you can't beat the close you have no edge).

Market-structure checks. CAVEAT: the workbook's odds are NOT Australian books
for most of history — Pinnacle 2014-Feb 2018, bet365 Sep 2018-Feb 2025, and
only from Sep 2025 an AU book (Betr). So these measure "opener vs close" for
a mostly-sharp book. The real AU-vs-sharp test is the FORWARD capture
(odds_snapshots / book_lines: Sportsbet + every AU book beside Pinnacle at
the same instant), which accumulates from the first poll.
  * MAE of the open vs the close (how much does the line sharpen?)
  * "follow the move": bet, at the OPEN number, the side the line later moved
    toward — how soft is the opener (upper bound: you can't know the move).
  * cover rate of the open vs the nflverse (US consensus) closing spread.

    python -m src.backtest            # prints + writes data/backtest_summary.json
"""
import json

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

from config import DATA_DIR, MARKET_BLEND, TRAIN_FROM_SEASON
from src.features import LINE_FREE_COLUMNS
from src.model import load_modelling_frame

BREAKEVEN = 1.0 / 1.909    # 52.4% at -110


def _odds(series, default=1.909):
    s = pd.to_numeric(series, errors="coerce")
    return s.where((s > 1.2) & (s < 3.5), default).fillna(default)


def walk_forward(df: pd.DataFrame, first_test: int = 2012) -> pd.DataFrame:
    rows = []
    for y in sorted(df["season"].unique()):
        if y < first_test:
            continue
        tr = df[(df["season"] < y) & (df["season"] >= TRAIN_FROM_SEASON)]
        te = df[df["season"] == y].copy()
        if len(tr) < 400 or not len(te):
            continue
        m = LinearRegression().fit(tr[LINE_FREE_COLUMNS], tr["home_margin"])
        te["pred_raw"] = m.predict(te[LINE_FREE_COLUMNS])
        te["pred_blend"] = (1 - MARKET_BLEND) * te["pred_raw"] + MARKET_BLEND * te["bookie_margin"]
        rows.append(te)
    return pd.concat(rows, ignore_index=True)


def simulate(te: pd.DataFrame, pred_col: str, line_col: str, threshold: float,
             settle_col: str = None) -> dict:
    """Bet home when pred - line >= t, away when <= -t; settle vs settle_col
    (default the same line) at that line's odds (fallback 1.909)."""
    settle_col = settle_col or line_col
    edge = te[pred_col] - te[line_col]
    side = np.where(edge >= threshold, 1, np.where(edge <= -threshold, -1, 0))
    mask = side != 0
    if not mask.any():
        return {"bets": 0}
    b = te[mask].copy()
    b["side"] = side[mask]
    # home covers if margin > -home_line  <=> margin - settle_margin > 0
    res = (b["home_margin"] - b[settle_col]) * b["side"]
    ho = _odds(b.get("Home Line Odds Open")); ao = _odds(b.get("Away Line Odds Open"))
    odds = np.where(b["side"] == 1, ho, ao)
    profit = np.where(res > 0, odds - 1.0, np.where(res < 0, -1.0, 0.0))
    dec = res != 0
    return {"bets": int(mask.sum()), "wins": int((res > 0).sum()), "losses": int((res < 0).sum()),
            "pushes": int((res == 0).sum()),
            "cover": float((res > 0).sum() / max(dec.sum(), 1)),
            "roi": float(profit.sum() / max(mask.sum(), 1))}


def market_structure(df: pd.DataFrame) -> dict:
    d = df.dropna(subset=["bookie_margin", "close_margin"]).copy()
    out = {"games": int(len(d)),
           "mae_open": float((d["home_margin"] - d["bookie_margin"]).abs().mean()),
           "mae_close": float((d["home_margin"] - d["close_margin"]).abs().mean())}
    # follow the move: at the OPEN number, back the side the close moved toward
    mv = d["close_margin"] - d["bookie_margin"]
    moved = d[mv.abs() >= 1.0].copy()
    side = np.sign(mv[mv.abs() >= 1.0])
    res = (moved["home_margin"] - moved["bookie_margin"]) * side
    dec = res != 0
    out["follow_move_bets"] = int(len(moved))
    out["follow_move_cover"] = float((res > 0).sum() / max(dec.sum(), 1))
    # vs the US consensus close (nflverse spread_line = home margin the market expects)
    if "nv_spread" in d.columns and d["nv_spread"].notna().any():
        u = d.dropna(subset=["nv_spread"])
        gap = u["nv_spread"] - u["bookie_margin"]           # US close - AU open
        big = u[gap.abs() >= 1.0]
        side = np.sign(gap[gap.abs() >= 1.0])
        res = (big["home_margin"] - big["bookie_margin"]) * side
        out["us_close_vs_au_open_games"] = int(len(big))
        out["us_close_vs_au_open_cover"] = float((res > 0).sum() / max((res != 0).sum(), 1))
        out["mae_us_close"] = float((u["home_margin"] - u["nv_spread"]).abs().mean())
        out["mae_au_close_same"] = float((u["home_margin"] - u["close_margin"]).abs().mean())
    return out


def rule_report() -> list:
    """The live signal rules vs the old ones, train 2012-18 / validate 2019-25."""
    from src.signals import handicap_signal, total_signal
    from src.totals import TOTAL_FEATURES
    df = load_modelling_frame()
    df = df[df["season"] <= int(df["season"].max())]
    out = []
    for y in range(2012, int(df["season"].max()) + 1):
        tr = df[(df["season"] < y) & (df["season"] >= TRAIN_FROM_SEASON)]; te = df[df["season"] == y].copy()
        if not len(te):
            continue
        m = LinearRegression().fit(tr[LINE_FREE_COLUMNS], tr["home_margin"])
        te["pred"] = (1 - MARKET_BLEND) * m.predict(te[LINE_FREE_COLUMNS]) + MARKET_BLEND * te["bookie_margin"]
        t_tr = tr.assign(exp_total=(tr.home_attack + tr.away_defense) / 2 + (tr.away_attack + tr.home_defense) / 2).dropna(subset=["exp_total", "total_points"])
        te["exp_total"] = (te.home_attack + te.away_defense) / 2 + (te.away_attack + te.home_defense) / 2
        te["total_open"] = pd.to_numeric(te.get("Total Score Open"), errors="coerce")
        ok = te["exp_total"].notna()
        tm = LinearRegression().fit(t_tr[TOTAL_FEATURES].fillna(0), t_tr["total_points"], sample_weight=0.85 ** (y - 1 - t_tr["season"]))
        te["pred_total"] = np.nan
        te.loc[ok, "pred_total"] = 0.5 * tm.predict(te.loc[ok, TOTAL_FEATURES].fillna(0)) + 0.5 * te.loc[ok, "total_open"]
        out.append(te)
    te = pd.concat(out)
    e = te["pred"] - te["bookie_margin"]
    hres = (te["home_margin"] - te["bookie_margin"]) * np.sign(e)
    te_t = te.dropna(subset=["pred_total", "total_open"])
    tres = (te_t["total_points"] - te_t["total_open"]) * np.sign(te_t["pred_total"] - te_t["total_open"])
    live_h = np.array([handicap_signal(a, b) is not None for a, b in zip(e, te["bookie_margin"])])
    live_t = (te_t["pred_total"] - te_t["total_open"]).map(total_signal).notna().to_numpy()
    rows = [("Handicap OLD |edge|>=2", hres, (e.abs() >= 2).to_numpy(), te["season"]),
            ("Handicap LIVE fav>=2 / dog>=3.5", hres, live_h, te["season"]),
            ("Totals OLD |edge|>=3 (no wind)", None, None, None),
            ("Totals LIVE under>=2 / over>=3 (wind)", tres, live_t, te_t["season"])]
    rep = []
    for name, res, mask, season in rows:
        if res is None:
            continue
        r = {"name": name}
        for lab, sm in (("train", season <= 2018), ("val", season >= 2019)):
            sel = res[mask & sm.to_numpy()]
            dec = sel[sel != 0]
            r[f"{lab}_n"] = int(len(sel))
            r[f"{lab}_cover"] = float((dec > 0).mean()) if len(dec) else 0.0
            r[f"{lab}_roi"] = float(((dec > 0).sum() * 0.909 - (dec < 0).sum()) / max(len(sel), 1))
        rep.append(r)
    return rep


def run(verbose: bool = True) -> dict:
    df = load_modelling_frame()
    wf = walk_forward(df)
    summary = {"seasons": [int(s) for s in sorted(wf["season"].unique())], "games": int(len(wf))}
    summary["mae"] = {
        "avg": float((wf["home_margin"] - wf["home_margin"].mean()).abs().mean()),
        "line_free": float((wf["home_margin"] - wf["pred_raw"]).abs().mean()),
        "blend": float((wf["home_margin"] - wf["pred_blend"]).abs().mean()),
        "book_open": float((wf["home_margin"] - wf["bookie_margin"]).abs().mean()),
        "book_close": float((wf["home_margin"] - wf["close_margin"]).abs().mean()),
    }
    sweep = {}
    for t in [1.0, 1.5, 2.0, 2.5, 3.0, 4.0]:
        sweep[str(t)] = {
            "vs_open": simulate(wf, "pred_blend", "bookie_margin", t),
            "vs_close": simulate(wf, "pred_blend", "bookie_margin", t, settle_col="close_margin"),
        }
    summary["edge_sweep"] = sweep
    summary["market"] = market_structure(wf)
    # per-season at 2.0 vs open
    summary["by_season"] = {int(s): simulate(g, "pred_blend", "bookie_margin", 2.0)
                            for s, g in wf.groupby("season")}
    try:
        summary["rules"] = rule_report()
    except Exception as ex:  # noqa: BLE001
        summary["rules_error"] = str(ex)
    summary["generated"] = pd.Timestamp.now().strftime("%Y-%m-%d %H:%M")
    try:
        (DATA_DIR / "backtest_summary.json").write_text(json.dumps(summary, indent=1, default=float))
    except Exception:  # noqa: BLE001
        pass
    if verbose:
        print(f"Walk-forward {summary['seasons'][0]}-{summary['seasons'][-1]}: {summary['games']} games")
        print("\nMAE (lower is better):")
        for k, v in summary["mae"].items():
            print(f"  {k:<12}{v:6.2f}")
        print(f"\nHandicap bets (blend vs workbook OPEN line), breakeven cover {BREAKEVEN:.1%}:")
        print(f"  {'edge>=':<8}{'bets':>6}{'cover(open)':>13}{'ROI(open)':>11}{'cover(close)':>14}{'ROI(close)':>12}")
        for t, r in sweep.items():
            o, c = r["vs_open"], r["vs_close"]
            print(f"  {t:<8}{o['bets']:>6}{o.get('cover', 0):>13.1%}{o.get('roi', 0):>11.1%}"
                  f"{c.get('cover', 0):>14.1%}{c.get('roi', 0):>12.1%}")
        for r in summary.get("rules", []):
            print(f"  RULE {r['name']:<40} train n={r['train_n']} cov {r['train_cover']:.1%} roi {r['train_roi']:+.1%} | "
                  f"validate n={r['val_n']} cov {r['val_cover']:.1%} roi {r['val_roi']:+.1%}")
        m = summary["market"]
        print("\nMarket structure (workbook book: Pinnacle->bet365->Betr; open vs close):")
        print(f"  MAE open {m['mae_open']:.2f}  close {m['mae_close']:.2f}  (n={m['games']})")
        print(f"  follow-the-move at the open (|move|>=1): {m['follow_move_bets']} bets, "
              f"cover {m['follow_move_cover']:.1%}")
        if "us_close_vs_au_open_cover" in m:
            print(f"  US close vs AU open (|gap|>=1): {m['us_close_vs_au_open_games']} games, "
                  f"backing the US-close side at the open covers {m['us_close_vs_au_open_cover']:.1%}")
            print(f"  MAE US close {m['mae_us_close']:.2f} vs workbook close {m['mae_au_close_same']:.2f}")
    return summary


if __name__ == "__main__":
    run()
