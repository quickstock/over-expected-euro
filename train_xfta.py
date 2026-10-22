"""Possession-level expected-FTA model (Poisson GLM), per league.

Same construction as the NBA v4 context model:
  - 7 features: period, seconds remaining, score margin at possession
    start, in_bonus (real FIBA penalty state here), offense_is_home,
    opponent-defense season foul rate (leave-one-game-out), officiating
    crew season rate (leave-one-game-out)
  - season cross-fit: each season predicted by a model trained on the
    other seasons; every prediction is out-of-fold
  - per-season anchoring on the finisher universe so league FTAOE sums
    to ~0 and seasons are comparable

Writes predictions_poss and the FTAOE leaderboard
player_season_xfta_lb.

Usage: python3 train_xfta.py [--league EL|ACB]
"""

import argparse
import sqlite3

import numpy as np
import pandas as pd
import statsmodels.api as sm

import config


def _logo_rate(df, key_cols):
    """Leave-one-game-out rate of sfta per possession for an entity
    (defense team or referee): (season_sum - game_sum) / (season_n -
    game_n), season-global fallback."""
    season_tot = df.groupby(key_cols + ["season"]).agg(
        s=("sfta", "sum"), n=("sfta", "size"))
    game_tot = df.groupby(key_cols + ["season", "game_id"]).agg(
        gs=("sfta", "sum"), gn=("sfta", "size"))
    j = game_tot.join(season_tot)
    # leave the game out; a zero denominator (entity's only game is this
    # one) must yield NaN so the season-mean fallback below fires
    denom = j.n - j.gn
    rate = (j.s - j.gs) / denom.where(denom > 0)
    season_mean = df.groupby("season").sfta.mean()
    rate = rate.reset_index().rename(columns={0: "rate"})
    rate["fallback"] = rate.season.map(season_mean)
    rate["rate"] = rate.rate.fillna(rate.fallback)
    return rate.drop(columns="fallback")


def build_features(con, league):
    poss = pd.read_sql(
        "SELECT league, season, game_id, possession_number, period, "
        "seconds_remaining_in_period, score_margin, in_bonus, "
        "offense_is_home, offense_team_id, sfta, finisher_player_id "
        "FROM possessions WHERE league = ?", con, params=(league,))
    games = pd.read_sql(
        "SELECT game_id, home_team_id, away_team_id, ref1, ref2, ref3 "
        "FROM games WHERE league = ?", con, params=(league,))
    df = poss.merge(games, on="game_id", how="left")

    # score margin at possession START: shift the terminal (home - away)
    # margin one possession back. Kept raw home-minus-away (per the NBA
    # contract); offense_is_home is a separate feature, so the GLM can
    # learn the home/away interaction rather than having it collapsed.
    df = df.sort_values(["game_id", "possession_number"])
    df["score_margin"] = (df.groupby("game_id").score_margin
                          .shift(1).fillna(0))

    df["def_team_id"] = np.where(df.offense_team_id == df.home_team_id,
                                 df.away_team_id, df.home_team_id)

    opp = _logo_rate(df.rename(columns={"def_team_id": "def_team"}),
                     ["def_team"]).rename(
        columns={"rate": "opp_rate_logo"})
    df = df.merge(opp, left_on=["def_team_id", "season", "game_id"],
                  right_on=["def_team", "season", "game_id"], how="left")

    refs = df.melt(
        id_vars=["game_id", "season", "sfta"],
        value_vars=["ref1", "ref2", "ref3"], value_name="ref")
    refs = refs[refs.ref.notna()]
    ref_rate = _logo_rate(refs, ["ref"]).rename(
        columns={"rate": "ref_rate"})
    refs = refs.merge(ref_rate, on=["ref", "season", "game_id"],
                      how="left")
    crew = (refs.groupby(["game_id"]).ref_rate.mean()
            .rename("crew_rate_logo").reset_index())
    df = df.merge(crew, on="game_id", how="left")

    season_mean = df.groupby("season").sfta.transform("mean")
    df["opp_rate_logo"] = df.opp_rate_logo.fillna(season_mean)
    df["crew_rate_logo"] = df.crew_rate_logo.fillna(season_mean)
    df["seconds_remaining_in_period"] = \
        df.seconds_remaining_in_period.fillna(0)
    return df


def cross_fit(df):
    """Season cross-fit Poisson GLM + per-season anchoring."""
    out = []
    for season in sorted(df.season.unique()):
        train = df[df.season != season]
        test = df[df.season == season].copy()
        X_tr = sm.add_constant(train[config.XFTA_FEATURES], has_constant="add")
        X_te = sm.add_constant(test[config.XFTA_FEATURES], has_constant="add")
        model = sm.GLM(train.sfta, X_tr,
                       family=sm.families.Poisson()).fit()
        test["raw_pred"] = model.predict(X_te)

        fin = test.finisher_player_id.notna()
        anchor = (test.loc[fin, "sfta"].mean()
                  / max(test.loc[fin, "raw_pred"].mean(), 1e-9))
        test["xfta"] = test.raw_pred * anchor

        d_pred = _poisson_deviance(test.loc[fin, "sfta"],
                                   test.loc[fin, "xfta"])
        base = np.full(fin.sum(), test.loc[fin, "sfta"].mean())
        d_base = _poisson_deviance(test.loc[fin, "sfta"], base)
        lift = (d_base - d_pred) / d_base * 100
        print(f"  {season}: n={len(test)} anchor={anchor:.4f} "
              f"deviance lift {lift:.2f}%")
        out.append(test)
    return pd.concat(out)


def _poisson_deviance(y, mu):
    mu = np.clip(np.asarray(mu, float), 1e-9, None)
    y = np.asarray(y, float)
    with np.errstate(divide="ignore", invalid="ignore"):
        term = np.where(y > 0, y * np.log(y / mu), 0.0)
    return 2 * np.sum(term - (y - mu))


def build_leaderboard(con, league, preds):
    """FTAOE board: actual drawn FTs minus the shot-diet expectation
    (diet_game, from build_diet_expected.py). The context predictions made
    here feed the team and referee views, not the player board."""
    lb = (preds[preds.finisher_player_id.notna()]
          .groupby(["season", "finisher_player_id"])
          .agg(possessions=("sfta", "size"),
               actual_fta_from_fouls=("sfta", "sum"))
          .reset_index()
          .rename(columns={"finisher_player_id": "player_id"}))
    diet = pd.read_sql(
        "SELECT season, player_id, SUM(xft) AS xfta_total FROM diet_game "
        "WHERE league = ? GROUP BY season, player_id", con, params=(league,))
    lb = lb.merge(diet, on=["season", "player_id"], how="left")
    lb["xfta_total"] = lb["xfta_total"].fillna(0.0)
    lb["ftaoe"] = lb.actual_fta_from_fouls - lb.xfta_total
    lb["ftaoe_per_100"] = lb.ftaoe / lb.possessions * 100
    lb["qualified"] = (lb.possessions >= config.QUALIFY_POSS).astype(int)
    lb["ftaoe_rank"] = (lb[lb.qualified == 1]
                        .groupby("season").ftaoe_per_100
                        .rank(ascending=False, method="min"))
    lb["league"] = league

    names = pd.read_sql(
        "SELECT season, player_id, player_name, position "
        "FROM player_season WHERE league = ?", con, params=(league,))
    lb = lb.merge(names, on=["season", "player_id"], how="left")
    return lb


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--league", choices=config.LEAGUES, default=None)
    args = ap.parse_args()
    leagues = [args.league] if args.league else config.LEAGUES

    con = sqlite3.connect(config.DB_PATH)
    for league in leagues:
        print(f"[{league}] building features")
        df = build_features(con, league)
        print(f"[{league}] cross-fitting {len(df)} possessions")
        preds = cross_fit(df)

        out = preds[["league", "season", "game_id", "possession_number",
                     "sfta", "xfta"]]
        _replace(con, "predictions_poss", league, out)

        lb = build_leaderboard(con, league, preds)
        _replace(con, "player_season_xfta_lb", league, lb)
        q = lb[lb.qualified == 1]
        print(f"[{league}] leaderboard: {len(q)} qualified player-seasons; "
              f"anchor check (poss-weighted mean/100): "
              f"{(q.ftaoe.sum() / q.possessions.sum() * 100):+.3f}")
    con.commit()
    con.close()


def _replace(con, name, league, df):
    if con.execute("SELECT 1 FROM sqlite_master WHERE name=?",
                   (name,)).fetchone():
        con.execute(f"DELETE FROM {name} WHERE league = ?", (league,))
    df.to_sql(name, con, if_exists="append", index=False)


if __name__ == "__main__":
    main()
