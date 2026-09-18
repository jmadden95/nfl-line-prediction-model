"""ntfy alerts: AU line moves, AU-vs-sharp lags (the thesis), and bet signals."""
import pandas as pd
import requests

from config import (ALERT_MOVE_THRESHOLD, LOCAL_TZ, NTFY_TOKEN, NTFY_TOPIC, NTFY_URL,
                    SHARP_LAG_MIN, SIGNAL_EDGE)
from src import db
from src.teams import short

_PHASE = {"open": "open (Tue)", "mid": "mid-week", "late": "Fri report", "pre_kick": "pre-kick",
          "kickoff": "kickoff", "page": "view"}


def push(title: str, message: str, priority: str = "default", tags: str = "football") -> bool:
    if not (NTFY_URL and NTFY_TOPIC):
        print(f"  (ntfy not configured) would alert: {title} — {message}")
        return False
    headers = {"Title": title.encode("ascii", "replace").decode("ascii"),
               "Priority": priority, "Tags": tags}
    if NTFY_TOKEN:
        headers["Authorization"] = f"Bearer {NTFY_TOKEN}"
    try:
        r = requests.post(f"{NTFY_URL}/{NTFY_TOPIC}", data=message.encode(), headers=headers, timeout=10)
        r.raise_for_status()
        return True
    except Exception as ex:  # noqa: BLE001
        print(f"  ntfy push failed: {ex}")
        return False


def _line(home, away, point) -> str:
    """home handicap point -> favourite phrase: -3.5 -> 'Chiefs -3.5'."""
    return f"{short(home)} {point:+.1f}" if point <= 0 else f"{short(away)} {-point:+.1f}"


def alert_line_moves(threshold: float = None) -> int:
    threshold = ALERT_MOVE_THRESHOLD if threshold is None else threshold
    snaps = db.odds_snapshots_df()
    if not len(snaps):
        return 0
    snaps = snaps[snaps["au_point"].notna()]
    today = pd.Timestamp.now(tz=LOCAL_TZ).date().isoformat()
    snaps = snaps[snaps["game_date"] >= today]
    sent = 0
    for (gd, home, away), g in snaps.groupby(["game_date", "home", "away"]):
        seq = []
        for _, s in g.sort_values("captured_ts").iterrows():
            v = float(s["au_point"])
            if seq and abs(seq[-1][1] - v) < 0.01:
                continue
            seq.append((s["phase"], v))
        if len(seq) < 2:
            continue
        (p0, v0), (p1, v1) = seq[-2], seq[-1]
        if abs(v1 - v0) < threshold:
            continue
        key = f"move|{gd}|{home}|{away}|{p1}|{v1:g}"
        if db.was_alerted(key):
            continue
        if push(f"NFL line move: {short(home)} v {short(away)}",
                f"{_PHASE.get(p0, p0)} {_line(home, away, v0)} -> {_PHASE.get(p1, p1)} "
                f"{_line(home, away, v1)} ({v1 - v0:+.1f})"):
            db.mark_alerted(key)
            sent += 1
    return sent


def alert_sharp_lags(events: list) -> int:
    """For each game: every AU book whose handicap trails the sharp line by
    >= SHARP_LAG_MIN. Positive (au_point - sharp_point) means the AU book gives
    the HOME side more points than the sharps do -> back HOME at that book."""
    sent = 0
    for e in events or []:
        sp = e.get("sharp_point")
        if sp is None or sp != sp:
            continue
        gd = str(pd.Timestamp(e["date"]).date())
        for b in e.get("books", []):
            if b.get("region") != "au" or b.get("sp_point") is None:
                continue
            lag = float(b["sp_point"]) - float(sp)
            if abs(lag) < SHARP_LAG_MIN:
                continue
            side = (f"{short(e['home'])} {float(b['sp_point']):+.1f}" if lag > 0
                    else f"{short(e['away'])} {-float(b['sp_point']):+.1f}")
            key = f"lag|{gd}|{e['home']}|{e['away']}|{b['key']}|{b['sp_point']:g}"
            if db.was_alerted(key):
                continue
            if push(f"AU soft line: {short(e['home'])} v {short(e['away'])}",
                    f"{b['title']} {_line(e['home'], e['away'], float(b['sp_point']))} vs sharp "
                    f"{_line(e['home'], e['away'], float(sp))} ({lag:+.1f}). Value: {side} @ {b['title']}.",
                    priority="4", tags="zap"):
                db.mark_alerted(key)
                sent += 1
    return sent


def alert_bet_signals(up) -> int:
    if up is None or not len(up):
        return 0
    sent = 0
    for _, r in up.iterrows():
        e = r.get("edge")
        if e is None or pd.isna(e) or abs(e) < SIGNAL_EDGE:
            continue
        home, away, bm = r["Home Team"], r["Away Team"], r["bookie_margin"]
        team, line = (home, -bm) if e > 0 else (away, bm)
        gd = str(pd.Timestamp(r["Date"]).date())
        key = f"betsig|{gd}|{home}|{away}|{'home' if e > 0 else 'away'}"
        if db.was_alerted(key):
            continue
        if push(f"NFL bet signal: {short(team)} {line:+.1f}",
                f"{home} v {away} ({gd}) — model edge {float(e):+.1f} vs {r.get('au_book')}. "
                f"BET {short(team)} {line:+.1f}.", tags="dart"):
            db.mark_alerted(key)
            sent += 1
    return sent
