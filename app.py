"""Streamlit browser for the Euro Over Expected results.

  streamlit run app.py

Leaderboard tab: league + season + metric, sortable table, name filter.
Player tab: search a player, see every season across both leagues.
"""

import sqlite3

import pandas as pd
import streamlit as st

import config

st.set_page_config(page_title="Euro Over Expected", layout="wide")

METRICS = {
    "Points over expected /100": "points_oe_per100",
    "FTA over expected /100": "ftaoe_per_100",
    "Shot-making (FG% - xFG%)": "shot_making_oe",
}

DISPLAY = {
    "season": "Season", "league": "Lg", "player_name": "Player",
    "position": "Pos", "possessions": "Poss", "fga": "FGA",
    "fg_pct": "FG%", "xfg_pct": "xFG%", "shot_making_oe": "ShotMake",
    "ftaoe_per_100": "FTAOE/100", "fg_pts_oe_per100": "FG ptsOE/100",
    "ft_pts_oe_per100": "FT ptsOE/100", "points_oe_per100": "PtsOE/100",
}


@st.cache_data
def load_sv():
    con = sqlite3.connect(config.DB_PATH)
    df = pd.read_sql("SELECT * FROM shot_value", con)
    con.close()
    df["fg_pct"] = (df.fg_pct * 100).round(1)
    df["xfg_pct"] = (df.xfg_pct * 100).round(1)
    for c in ("shot_making_oe", "ftaoe_per_100", "fg_pts_oe_per100",
              "ft_pts_oe_per100", "points_oe_per100"):
        df[c] = df[c].round(2)
    return df


def show(df, sort_col):
    cols = list(DISPLAY)
    out = (df.sort_values(sort_col, ascending=False)[cols]
             .rename(columns=DISPLAY)
             .reset_index(drop=True))
    out.index += 1
    st.dataframe(out, use_container_width=True, height=650)


sv = load_sv()
tab_lb, tab_player = st.tabs(["Leaderboard", "Player"])

with tab_lb:
    c1, c2, c3, c4 = st.columns([1, 1, 2, 2])
    league = c1.selectbox("League", config.LEAGUES)
    seasons = sorted(sv[sv.league == league].season.unique(),
                     reverse=True)
    season = c2.selectbox("Season", seasons)
    metric = c3.selectbox("Metric", list(METRICS))
    name = c4.text_input("Filter by name", "")

    df = sv[(sv.league == league) & (sv.season == season)
            & (sv.qualified == 1)]
    if name:
        df = df[df.player_name.str.contains(name, case=False, na=False)]
    st.caption(f"{len(df)} qualified player-seasons "
               f"(≥{config.QUALIFY_POSS} finisher-possessions)")
    show(df, METRICS[metric])

with tab_player:
    q = st.text_input("Player search", "")
    if q:
        hits = sv[sv.player_name.str.contains(q, case=False, na=False)]
        if hits.empty:
            st.info("No player found.")
        else:
            for pname, g in hits.groupby("player_name"):
                st.subheader(pname)
                show(g.sort_values("season"), "points_oe_per100")
    else:
        st.caption("Type a name — results cover EuroLeague and Liga ACB, "
                   "2020-21 to 2025-26, including non-qualified seasons.")
