"""Bet settlement, performance tally, and the unbiased signal scorecards."""
import numpy as np
import pandas as pd

from config import SHARP_LAG_MIN, SIGNAL_EDGE, TOTAL_LAG_MIN, TOTAL_SIGNAL
from src import db

STD_ODDS = 1.909


def settle_bets(bets: pd.DataFrame) -> pd.DataFrame:
    """Fill result/profit from known scores; close_line = pick side's line at the
    LAST captured AU snapshot (kickoff phase when we have it)."""
    if bets is None or not len(bets):
        return bets
    open_mask = ~bets["result"].isin(["W", "L", "P"])
    if not open_mask.any():
        return bets
    from src.load_data import load_raw
    try:
        m = load_raw().dropna(subset=["Home Score", "Away Score"])
    except Exception:  # noqa: BLE001
        return bets
    key = {(str(d.date()), h, a): (hs, as_) for d, h, a, hs, as_ in
           zip(m["Date"], m["Home Team"], m["Away Team"], m["Home Score"], m["Away Score"])}
    snaps = db.odds_snapshots_df()
    for i, b in bets[open_mask].iterrows():
        gd = str(b["gdate"])[:10]
        is_total = b["side"] in ("over", "under")
        close = None
        if len(snaps):
            k = snaps[(snaps["game_date"] == gd) & (snaps["home"] == b["home"]) & (snaps["away"] == b["away"])]
            col = "au_total" if is_total else "au_point"
            k = k[k[col].notna()].sort_values("captured_ts")
            if len(k):
                cp = float(k.iloc[-1][col])
                close = cp if is_total else (cp if b["side"] == "home" else -cp)
        sc = key.get((gd, b["home"], b["away"]))
        if sc is None:
            if close is not None and pd.isna(b.get("close_line")):
                db.update_bet(int(b["rowid"]), close_line=close)
                bets.loc[i, "close_line"] = close
            continue
        hs, as_ = sc
        if is_total:
            res = (hs + as_) - float(b["line"]) if b["side"] == "over" else float(b["line"]) - (hs + as_)
        else:
            margin = hs - as_ if b["side"] == "home" else as_ - hs
            res = margin + float(b["line"])
        result = "W" if res > 0 else ("L" if res < 0 else "P")
        odds, stake = float(b["odds"] or STD_ODDS), float(b["stake"] or 0)
        profit = stake * (odds - 1) if result == "W" else (-stake if result == "L" else 0.0)
        db.update_bet(int(b["rowid"]), home_score=hs, away_score=as_, result=result, profit=profit, close_line=close)
        bets.loc[i, ["home_score", "away_score", "result", "profit", "close_line"]] = [hs, as_, result, profit, close]
    return bets


def bet_clv(g: pd.DataFrame) -> pd.Series:
    """Closing-line value per bet (pts, + = you beat the close). Handicap: line - close.
    Over: close - line (line went up = you got the low number). Under: line - close."""
    line = pd.to_numeric(g["line"], errors="coerce"); close = pd.to_numeric(g["close_line"], errors="coerce")
    raw = line - close
    return raw.where(g["side"] != "over", close - line)


def tally(bets: pd.DataFrame) -> list:
    """Performance by bet source (model / soft / manual) + ALL."""
    if bets is None or not len(bets):
        return []
    b = bets.copy()
    b["model"] = b["model"].fillna("manual").replace("", "manual")
    rows = []
    groups = list(b.groupby("model")) + [("ALL", b)]
    for name, g in groups:
        s = g[g["result"].isin(["W", "L", "P"])]
        staked = pd.to_numeric(s["stake"], errors="coerce").fillna(0).sum()
        pl = pd.to_numeric(s["profit"], errors="coerce").fillna(0).sum()
        clv = bet_clv(g).dropna()
        rows.append({"label": name, "n": len(g), "settled": len(s),
                     "w": int((s["result"] == "W").sum()), "l": int((s["result"] == "L").sum()),
                     "p": int((s["result"] == "P").sum()), "pl": pl,
                     "roi": (pl / staked) if staked else 0.0,
                     "clv": f"{clv.mean():+.2f}" if len(clv) else "–",
                     "beat": f"{(clv > 0).mean():.0%}" if len(clv) else "–"})
    return rows


def _results_index():
    """(gdate, home, away) -> (home margin, total points)."""
    from src.load_data import load_raw
    try:
        m = load_raw().dropna(subset=["Home Score", "Away Score"])
    except Exception:  # noqa: BLE001
        return {}
    return {(str(d.date()), h, a): (hs - as_, hs + as_) for d, h, a, hs, as_ in
            zip(m["Date"], m["Home Team"], m["Away Team"], m["Home Score"], m["Away Score"])}


def _score(recs: list, label: str) -> dict | None:
    if not recs:
        return None
    d = pd.DataFrame(recs)
    out = {"label": label, "n": len(d), "avg_clv": f"{d['clv'].mean():+.2f}",
           "beat": f"{(d['clv'] > 0).mean():.0%}", "pos": d["clv"].mean() > 0}
    r = d.dropna(subset=["cover"])
    dec = r[r["cover"] != 0]
    if len(dec):
        wins = (dec["cover"] > 0).sum()
        out["settled"] = len(dec); out["cover"] = f"{wins / len(dec):.0%}"
        out["roi"] = f"{(wins * (STD_ODDS - 1) - (len(dec) - wins)) / len(dec):+.1%}"
    return out


def signal_scorecards() -> list:
    """Unbiased forward performance of EVERY flagged signal (taken or not).
    Model signal: first daily prediction capture with |edge| >= SIGNAL_EDGE; CLV =
    pick-side AU line at that capture vs the last AU capture; cover from results.
    AU-soft signal: first snapshot where the AU book trails the sharp line by
    >= SHARP_LAG_MIN; CLV = AU line then vs AU close (did Sportsbet move to the
    sharp number?); cover at the soft line from results."""
    preds = db.predictions_df()
    snaps = db.odds_snapshots_df()
    results = _results_index()
    if not len(snaps):
        return []
    snaps = snaps[snaps["au_point"].notna()].sort_values("captured_ts")
    last_au, last_tot = {}, {}
    for (gd, h, a), g in snaps.groupby(["game_date", "home", "away"]):
        last_au[(gd, h, a)] = float(g.iloc[-1]["au_point"])
        gt = g[g["au_total"].notna()]
        if len(gt):
            last_tot[(gd, h, a)] = float(gt.iloc[-1]["au_total"])
    model_recs, soft_recs, total_recs, soft_tot_recs = [], [], [], []
    if len(preds):
        preds = preds.sort_values("captured_ts")
        for (gd, h, a), g in preds.groupby(["game_date", "home", "away"]):
            first = g[g["edge"].abs() >= SIGNAL_EDGE].head(1)
            if not len(first) or pd.isna(first.iloc[0]["au_point"]):
                continue
            f = first.iloc[0]
            side = 1 if f["edge"] > 0 else -1                      # +1 home
            pick_line = float(f["au_point"]) * side               # pick side's handicap
            close = last_au.get((gd, h, a))
            if close is None:
                continue
            clv = pick_line - close * side
            mg = results.get((gd, h, a))
            cover = (mg[0] * side + pick_line) if mg is not None else np.nan
            model_recs.append({"clv": clv, "cover": cover})
        # totals: first capture with |pred_total - au_total| >= TOTAL_SIGNAL
        if "pred_total" in preds.columns:
            for (gd, h, a), g in preds.groupby(["game_date", "home", "away"]):
                te = pd.to_numeric(g["pred_total"], errors="coerce") - pd.to_numeric(g["au_total"], errors="coerce")
                first = g[te.abs() >= TOTAL_SIGNAL].head(1)
                if not len(first):
                    continue
                f = first.iloc[0]
                over = float(f["pred_total"]) > float(f["au_total"])
                line = float(f["au_total"])
                close_t = last_tot.get((gd, h, a))
                if close_t is None:
                    continue
                clv = (close_t - line) if over else (line - close_t)
                mg = results.get((gd, h, a))
                cover = ((mg[1] - line) if over else (line - mg[1])) if mg is not None else np.nan
                total_recs.append({"clv": clv, "cover": cover})
    for (gd, h, a), g in snaps.groupby(["game_date", "home", "away"]):
        gs = g[g["sharp_point"].notna()]
        gs = gs[(gs["au_point"] - gs["sharp_point"]).abs() >= SHARP_LAG_MIN]
        if not len(gs) or g["captured_ts"].nunique() < 2:
            continue
        f = gs.iloc[0]
        gap = float(f["au_point"]) - float(f["sharp_point"])
        side = 1 if gap > 0 else -1                                # AU gives home more points -> home
        pick_line = float(f["au_point"]) * side
        close = last_au[(gd, h, a)]
        clv = pick_line - close * side
        mg = results.get((gd, h, a))
        cover = (mg[0] * side + pick_line) if mg is not None else np.nan
        soft_recs.append({"clv": clv, "cover": cover})
    # soft totals: AU total vs sharp total gap >= TOTAL_LAG_MIN (AU higher -> UNDER at AU)
    for (gd, h, a), g in snaps.groupby(["game_date", "home", "away"]):
        gs = g[g["sharp_total"].notna() & g["au_total"].notna()]
        gs = gs[(gs["au_total"] - gs["sharp_total"]).abs() >= TOTAL_LAG_MIN]
        if not len(gs) or (gd, h, a) not in last_tot or g["captured_ts"].nunique() < 2:
            continue
        f = gs.iloc[0]
        under = float(f["au_total"]) > float(f["sharp_total"])
        line = float(f["au_total"]); close_t = last_tot[(gd, h, a)]
        clv = (line - close_t) if under else (close_t - line)
        mg = results.get((gd, h, a))
        cover = ((line - mg[1]) if under else (mg[1] - line)) if mg is not None else np.nan
        soft_tot_recs.append({"clv": clv, "cover": cover})
    return [s for s in (_score(model_recs, "Handicap: model (|edge| ≥ %.1f)" % SIGNAL_EDGE),
                        _score(soft_recs, "Handicap: AU soft (trails sharp ≥ %.1f)" % SHARP_LAG_MIN),
                        _score(total_recs, "Totals: model (|edge| ≥ %.1f)" % TOTAL_SIGNAL),
                        _score(soft_tot_recs, "Totals: AU soft (vs sharp ≥ %.1f)" % TOTAL_LAG_MIN)) if s]
