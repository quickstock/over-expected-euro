"""CLI search over the Over Expected results.

  python3 search.py micic                 # find a player, all seasons
  python3 search.py --league ACB --season 2024-25 --top 20
  python3 search.py --league EL --season 2024-25 --sort shot_making --top 15

Sort keys: points_oe (default), ftaoe, shot_making.
"""

import argparse
import sqlite3

import pandas as pd

import config

SORT_COL = {"points_oe": "points_oe_per100",
            "ftaoe": "ftaoe_per_100",
            "shot_making": "shot_making_oe"}

COLS = ["league", "season", "player_name", "position", "possessions",
        "fga", "fg_pct", "xfg_pct", "shot_making_oe", "ftaoe_per_100",
        "fg_pts_oe_per100", "ft_pts_oe_per100", "points_oe_per100"]


def load(con, league=None, season=None, name=None, qualified=True):
    q = "SELECT * FROM shot_value WHERE 1=1"
    params = []
    if league:
        q += " AND league = ?"
        params.append(league)
    if season:
        q += " AND season = ?"
        params.append(season)
    if name:
        q += " AND player_name LIKE ?"
        params.append(f"%{name}%")
    if qualified:
        q += " AND qualified = 1"
    return pd.read_sql(q, con, params=params)


def fmt(df, sort_col):
    df = df.sort_values(sort_col, ascending=False)[COLS].copy()
    df["fg_pct"] = (df.fg_pct * 100).round(1)
    df["xfg_pct"] = (df.xfg_pct * 100).round(1)
    for c in ("shot_making_oe", "ftaoe_per_100", "fg_pts_oe_per100",
              "ft_pts_oe_per100", "points_oe_per100"):
        df[c] = df[c].round(2)
    df.columns = ["Lg", "Season", "Player", "Pos", "Poss", "FGA", "FG%",
                  "xFG%", "ShotMake", "FTAOE/100", "FGptsOE", "FTptsOE",
                  "PtsOE/100"]
    return df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("name", nargs="?", default=None,
                    help="player name substring")
    ap.add_argument("--league", choices=config.LEAGUES)
    ap.add_argument("--season", choices=config.SEASONS)
    ap.add_argument("--sort", choices=list(SORT_COL), default="points_oe")
    ap.add_argument("--top", type=int, default=25)
    ap.add_argument("--all", action="store_true",
                    help="include non-qualified player-seasons")
    args = ap.parse_args()

    con = sqlite3.connect(config.DB_PATH)
    df = load(con, args.league, args.season, args.name,
              qualified=not (args.all or args.name))
    con.close()
    if df.empty:
        print("no results")
        return
    out = fmt(df, SORT_COL[args.sort]).head(args.top)
    print(out.to_string(index=False))


if __name__ == "__main__":
    main()
