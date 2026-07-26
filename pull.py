"""Network pull -> cache/. No table building here.

Usage:
  python3 pull.py --league EL [--season 2023-24]
  python3 pull.py --league ACB

Idempotent: cached games are skipped, so re-running fills gaps after
failures. Failures are logged and skipped (pull again to retry).
"""

import argparse
import sys
import time

import config
from leagues import ADAPTERS


def pull_league(league, seasons):
    adapter = ADAPTERS[league]
    failures = []
    for season in seasons:
        try:
            games = adapter.list_games(season)
            if league == "EL":
                adapter.pull_people(season)
        except Exception as e:  # season-list fetch failed; skip, retry later
            failures.append((league, season, "<season-list>", repr(e)))
            print(f"[{league} {season}] LIST FAIL: {e}", flush=True)
            continue
        print(f"[{league} {season}] {len(games)} regular-season games",
              flush=True)
        t0 = time.time()
        for i, g in enumerate(games, 1):
            try:
                adapter.pull_game(season, g["game_code"])
            except Exception as e:  # log-and-continue: re-run to retry
                failures.append((league, season, g["game_id"], repr(e)))
                print(f"  FAIL {g['game_id']}: {e}", flush=True)
            if i % 50 == 0:
                rate = i / max(time.time() - t0, 1)
                print(f"  {i}/{len(games)} ({rate:.1f} games/s)", flush=True)
        print(f"[{league} {season}] done in {time.time()-t0:.0f}s",
              flush=True)
    return failures


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--league", choices=list(ADAPTERS), required=True)
    ap.add_argument("--season", choices=config.SEASONS, default=None)
    args = ap.parse_args()

    seasons = [args.season] if args.season else config.SEASONS
    failures = pull_league(args.league, seasons)
    if failures:
        print(f"\n{len(failures)} failures:")
        for f in failures:
            print(" ", *f)
        sys.exit(1)
    print("\npull complete, no failures")


if __name__ == "__main__":
    main()
