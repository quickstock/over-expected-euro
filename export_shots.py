"""A sample of each season's shots -> site/public/shots-{LG}-{season}.json.

Same construction as ../NBA_OE/export_shots.py, per league: a fixed-seed
sample of SAMPLE half-court shots per season, in centimetres from the hoop,
for the homepage to draw behind its title. Seasons are each league's exported
core's own, so the site never asks for a season it does not list.
"""
import json
import zlib

import numpy as np

import config
from export_site_data import OUT_DIR, conn

SAMPLE = 6000
HALF_COURT_M = 14.0


def sample(league: str, season: str) -> dict:
    rows = np.array(conn.execute(
        """SELECT x_m, y_m, shot_made FROM shots
           WHERE league = ? AND season = ? AND y_m < ?""",
        (league, season, HALF_COURT_M)).fetchall(), dtype=float)
    # A season a few games old has fewer shots than the sample: draw them all.
    rng = np.random.default_rng(zlib.crc32(f"{league}{season}".encode()))
    pick = rows[rng.choice(len(rows), min(SAMPLE, len(rows)), replace=False)]
    return {
        "x": np.rint(pick[:, 0] * 100).astype(int).tolist(),
        "y": np.rint(pick[:, 1] * 100).astype(int).tolist(),
        "m": "".join("1" if m else "0" for m in pick[:, 2]),
    }


def main() -> None:
    for league in config.LEAGUES:
        seasons = json.loads((OUT_DIR / f"data-{league}.json").read_text())["meta"]["seasons"]
        for s in seasons:
            (OUT_DIR / f"shots-{league}-{s}.json").write_text(
                json.dumps(sample(league, s), separators=(",", ":")))
        print(f"shots {league}: {len(seasons)} seasons x {SAMPLE}")


if __name__ == "__main__":
    main()
