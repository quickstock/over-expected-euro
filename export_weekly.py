"""The week's best and worst shooting games -> site/public/weekly-{LG}.json.

Same construction as ../NBA_OE/export_weekly.py, per league: player-games
ranked by field-goal points over expected (player_game_shot_value), Monday to
Sunday weeks keyed by their Monday, finished weeks only, from 2025-26 on, top 3
each way. For the WNBA "2025-26" is its 2025 season, so its weeks start in May.

GBL is left out: its game ids come from a Wayback manifest and its feed carries
no date, so its game dates are interpolated across the season (leagues/gbl.py)
and cannot place a game in a week.
"""
import datetime as dt
import json

import pandas as pd

import config
from export_site_data import OUT_DIR, conn, team_abbrev_map

FROM_SEASON = "2025-26"
TOP = 3
LEAGUES = [lg for lg in config.LEAGUES if lg != "GBL"]


def export(league):
    g = pd.read_sql(
        """SELECT p.player_id, p.game_id, p.season, p.fga, p.fgm,
                  p.act_fg_pts, p.exp_fg_pts, g.date,
                  g.home_team_id, g.away_team_id
           FROM player_game_shot_value p
           JOIN games g ON g.league = p.league AND g.game_id = p.game_id
           WHERE p.league = ? AND p.season >= ?""",
        conn, params=(league, FROM_SEASON))
    team = pd.read_sql(
        """SELECT game_id, player_id, team_id, COUNT(*) AS n FROM shots
           WHERE league = ? AND season >= ? AND team_id IS NOT NULL
           GROUP BY game_id, player_id, team_id""",
        conn, params=(league, FROM_SEASON))
    team = team.sort_values("n").drop_duplicates(["game_id", "player_id"],
                                                 keep="last")
    names = pd.read_sql(
        "SELECT player_id, season, player_name FROM player_season "
        "WHERE league = ? AND season >= ?", conn,
        params=(league, FROM_SEASON)).drop_duplicates(["player_id", "season"])
    abbrev = team_abbrev_map(league)

    g = g.merge(team[["game_id", "player_id", "team_id"]],
                on=["game_id", "player_id"], how="inner")
    g = g.merge(names, on=["player_id", "season"], how="left")
    g["date"] = pd.to_datetime(g["date"]).dt.date
    g["week"] = g["date"].map(lambda d: d - dt.timedelta(days=d.weekday()))
    g = g[g["week"] + dt.timedelta(days=6) < dt.date.today()]
    g["home"] = g["team_id"] == g["home_team_id"]
    g["opp_id"] = g["away_team_id"].where(g["home"], g["home_team_id"])
    g["oe"] = g["act_fg_pts"] - g["exp_fg_pts"]

    def row(r):
        return {"id": str(r.player_id), "name": r.player_name,
                "team": abbrev.get(r.team_id, r.team_id),
                "opp": abbrev.get(r.opp_id, r.opp_id), "home": bool(r.home),
                "date": r.date.isoformat(), "fgm": int(r.fgm),
                "fga": int(r.fga), "pts": int(r.act_fg_pts),
                "xpts": round(float(r.exp_fg_pts), 1),
                "oe": round(float(r.oe), 1),
                "gid": str(r.game_id), "season": r.season}

    weeks = {}
    for week, w in g.groupby("week"):
        w = w.sort_values("oe", ascending=False)
        weeks[week.isoformat()] = {
            "games": int(w["game_id"].nunique()),
            "over": [row(r) for r in w.head(TOP).itertuples()],
            "under": [row(r) for r in w.tail(TOP)[::-1].itertuples()],
        }
    out = OUT_DIR / f"weekly-{league}.json"
    out.write_text(json.dumps({"league": league, "weeks": weeks},
                              separators=(",", ":"), ensure_ascii=False))
    print(f"{out.name}: {len(weeks)} weeks, latest "
          f"{max(weeks) if weeks else '-'}")


if __name__ == "__main__":
    for lg in LEAGUES:
        export(lg)
