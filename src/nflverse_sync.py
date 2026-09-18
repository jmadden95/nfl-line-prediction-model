"""Refresh the cached nflverse games.csv (schedule, results, rest, roof, QBs).

    python -m src.nflverse_sync
"""
import requests

from config import NFLVERSE_GAMES_CSV, NFLVERSE_GAMES_URL


def refresh() -> int:
    r = requests.get(NFLVERSE_GAMES_URL, timeout=60)
    r.raise_for_status()
    if len(r.content) < 100_000 or not r.content.startswith(b"game_id"):
        raise RuntimeError("nflverse games.csv looks wrong; not overwriting")
    tmp = NFLVERSE_GAMES_CSV.with_suffix(".tmp")
    tmp.write_bytes(r.content)
    tmp.replace(NFLVERSE_GAMES_CSV)
    return len(r.content)


if __name__ == "__main__":
    print(f"nflverse: wrote {refresh()} bytes to {NFLVERSE_GAMES_CSV}")
