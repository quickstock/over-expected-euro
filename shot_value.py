"""xFG% (LightGBM, leak-free season cross-fit) + shot-value suite.

Same construction as the NBA system:
  - xFG%: shot-context-only make probability. Never sees the shooter.
    OOF by season cross-fit, anchored per season so league FG points
    over expected sits at zero.
  - run_margin = |score_margin - pts*shot_made| corrects the post-event
    stored margin back to pre-shot.
  - Shot value: expected points per shot = make value (xFG * pts) plus
    foul value (xFTA * league FT%); the headline points_oe_per100
    credits actual conversion on both sides (FGs at the player's rate,
    drawn FTs at his own FT%).

Writes: shots_xfg, shot_value, player_game_shot_value, team_shot_value.

Usage: python3 shot_value.py [--league EL|ACB]
"""

import argparse
import sqlite3

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier

import config


def make_clf():
    return LGBMClassifier(**config.LGBM_PARAMS, verbose=-1)


def load_shots(con, league):
    # second_chance / off_turnover come from possession_shot_flags
    # (derived from possession structure, leak-free); dunk from shots.
    df = pd.read_sql(
        "SELECT s.league, s.season, s.game_id, s.event_id, s.player_id, "
        "s.team_id, s.period, "
        "s.secs_remaining AS seconds_remaining_in_period, "
        "s.shot_made, s.shot_value, s.x_m, s.y_m, s.distance_m, "
        "s.angle_deg, s.zone_basic, s.zone_area, s.shot_type, "
        "s.score_margin, s.home_or_away, s.dunk, "
        "COALESCE(f.second_chance, 0) AS second_chance, "
        "COALESCE(f.off_turnover, 0) AS off_turnover "
        "FROM shots s LEFT JOIN possession_shot_flags f "
        "ON f.league = s.league AND f.game_id = s.game_id "
        "AND f.event_id = s.event_id "
        "WHERE s.league = ?", con, params=(league,))
    df["seconds_remaining_in_period"] = \
        df.seconds_remaining_in_period.fillna(0)
    df["run_margin"] = (df.score_margin
                        - df.shot_value * df.shot_made).abs()
    for c in config.XFG_CAT_FEATURES:
        df[c] = df[c].astype("category")
    return df


def cross_fit_xfg(df, league):
    features = (config.XFG_NUM_FEATURES + config.XFG_FLAGS[league]
                + config.XFG_CAT_FEATURES)
    out = []
    for season in sorted(df.season.unique()):
        train, test = df[df.season != season], df[df.season == season].copy()
        clf = make_clf()
        clf.fit(train[features], train.shot_made,
                categorical_feature=config.XFG_CAT_FEATURES)
        test["xfg"] = clf.predict_proba(test[features])[:, 1]
        anchor = test.shot_made.mean() / max(test.xfg.mean(), 1e-9)
        test["xfg"] *= anchor
        print(f"  {season}: n={len(test)} anchor={anchor:.4f} "
              f"fg%={test.shot_made.mean():.3f}")
        out.append(test)
    return pd.concat(out)


def aggregate(con, league, df):
    df["exp_fg_pts"] = df.xfg * df.shot_value
    df["act_fg_pts"] = df.shot_made * df.shot_value

    ps = (df.groupby(["season", "player_id"])
            .agg(fga=("shot_made", "size"), fgm=("shot_made", "sum"),
                 xfg_sum=("xfg", "sum"),
                 exp_fg_pts=("exp_fg_pts", "sum"),
                 act_fg_pts=("act_fg_pts", "sum"))
            .reset_index())

    lb = pd.read_sql(
        "SELECT season, player_id, player_name, position, possessions, "
        "actual_fta_from_fouls AS actual_fta, xfta_total, ftaoe, "
        "qualified FROM player_season_xfta_lb WHERE league = ?",
        con, params=(league,))
    ft = pd.read_sql(
        "SELECT season, player_id, ftm, fta, ft_pct "
        "FROM player_season_ft WHERE league = ?", con, params=(league,))
    lf = ft.groupby("season")[["ftm", "fta"]].sum()
    league_ft = (lf.ftm / lf.fta.clip(lower=1)) \
        .rename("league_ft").reset_index()

    sv = ps.merge(lb, on=["season", "player_id"], how="inner") \
           .merge(ft[["season", "player_id", "ft_pct"]],
                  on=["season", "player_id"], how="left") \
           .merge(league_ft, on="season", how="left")
    sv["ft_pct"] = np.where(sv.ft_pct.fillna(0) > 0, sv.ft_pct,
                            sv.league_ft)

    sv["fg_pct"] = sv.fgm / sv.fga
    sv["xfg_pct"] = sv.xfg_sum / sv.fga
    sv["shot_making_oe"] = (sv.fg_pct - sv.xfg_pct) * 100
    sv["xpoints_per_shot"] = (sv.exp_fg_pts
                              + sv.xfta_total * sv.league_ft) / sv.fga
    sv["fg_pts_oe"] = sv.act_fg_pts - sv.exp_fg_pts
    sv["ft_pts_oe"] = (sv.actual_fta * sv.ft_pct
                       - sv.xfta_total * sv.league_ft)
    sv["points_oe"] = sv.fg_pts_oe + sv.ft_pts_oe
    for c in ("fg_pts_oe", "ft_pts_oe", "points_oe"):
        sv[f"{c}_per100"] = sv[c] / sv.possessions * 100
    sv["ftaoe_per_100"] = sv.ftaoe / sv.possessions * 100
    sv["league"] = league

    pg = (df.groupby(["season", "player_id", "game_id"])
            .agg(fga=("shot_made", "size"), fgm=("shot_made", "sum"),
                 act_fg_pts=("act_fg_pts", "sum"),
                 exp_fg_pts=("exp_fg_pts", "sum"))
            .reset_index())
    pg["league"] = league

    games = pd.read_sql(
        "SELECT game_id, home_team_id, away_team_id FROM games "
        "WHERE league = ?", con, params=(league,))
    tg = (df.groupby(["season", "team_id", "game_id"])
            .agg(fga=("shot_made", "size"),
                 act_fg_pts=("act_fg_pts", "sum"),
                 exp_fg_pts=("exp_fg_pts", "sum"))
            .reset_index()
            .merge(games, on="game_id"))
    tg["def_team_id"] = np.where(tg.team_id == tg.home_team_id,
                                 tg.away_team_id, tg.home_team_id)
    off = tg.groupby(["season", "team_id"]).agg(
        off_fga=("fga", "sum"), off_act_fg_pts=("act_fg_pts", "sum"),
        off_exp_fg_pts=("exp_fg_pts", "sum"))
    dfn = tg.groupby(["season", "def_team_id"]).agg(
        def_fga=("fga", "sum"), def_act_fg_pts=("act_fg_pts", "sum"),
        def_exp_fg_pts=("exp_fg_pts", "sum"))
    dfn.index.names = ["season", "team_id"]
    tsv = off.join(dfn).reset_index()
    tsv["league"] = league
    return sv, pg, tsv


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--league", choices=config.LEAGUES, default=None)
    args = ap.parse_args()
    leagues = [args.league] if args.league else config.LEAGUES

    con = sqlite3.connect(config.DB_PATH)
    for league in leagues:
        df = load_shots(con, league)
        print(f"[{league}] {len(df)} shots, cross-fitting xFG")
        df = cross_fit_xfg(df, league)

        _replace(con, "shots_xfg",
                 league, df[["league", "season", "game_id", "event_id",
                             "xfg"]])
        sv, pg, tsv = aggregate(con, league, df)
        _replace(con, "shot_value", league, sv)
        _replace(con, "player_game_shot_value", league, pg)
        _replace(con, "team_shot_value", league, tsv)
        q = sv[sv.qualified == 1]
        print(f"[{league}] shot_value: {len(q)} qualified player-seasons; "
              f"points_oe anchor check: "
              f"{(q.points_oe.sum() / q.possessions.sum() * 100):+.3f}/100")
    con.commit()
    con.close()


def _replace(con, name, league, df):
    if con.execute("SELECT 1 FROM sqlite_master WHERE name=?",
                   (name,)).fetchone():
        con.execute(f"DELETE FROM {name} WHERE league = ?", (league,))
    df.to_sql(name, con, if_exists="append", index=False)


if __name__ == "__main__":
    main()
