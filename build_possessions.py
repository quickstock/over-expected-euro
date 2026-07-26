"""events -> possessions (+ and1_shots, per-shot derived flags).

Port of the NBA possession contract:
  - offense per event by running state (shots/FTs/turnovers set it,
    defensive rebounds flip it, fouls never change it)
  - possession cut when offense changes between consecutive events
  - sfta = FTs from personal fouls (and-1s, 2-FT and 3-FT trips).
    FIBA PBP cannot separate shooting fouls from bonus-penalty fouls, so
    unlike the NBA build, penalty FTs are IN — the stat reads "free
    throws drawn". Technical / unsportsmanlike / bench FTs are excluded,
    broken or ambiguous trips are excluded and counted.
  - finisher = FT shooter, else last made-FG shooter, else last
    turnover, else last FGA shooter
  - in_bonus is the real FIBA team-foul penalty state (defense has >= 4
    counted fouls in the quarter; OT extends Q4), computed at the start
    of the possession — an upgrade over the NBA period>=4 proxy.

Usage: python3 build_possessions.py [--league EL|ACB]
"""

import argparse
import sqlite3
from collections import defaultdict

import pandas as pd

import config
from leagues.common import COUNTED_FOUL_KINDS, EXCLUDED_TRIP_FOUL_KINDS

OFFENSE_SETTERS = {"fg", "ft", "turnover"}


def compute_offense(events):
    """Running offense_team_id per event; leading unknowns backfilled."""
    cur, out = None, []
    for e in events:
        t = e["team_id"]
        if e["etype"] in OFFENSE_SETTERS and t:
            cur = t
        elif e["etype"] == "reb_def" and t and t != cur:
            cur = t
        out.append(cur)
    first = next((o for o in out if o is not None), None)
    return [o if o is not None else first for o in out]


def split_possessions(events, offense):
    groups, start = [], 0
    for i in range(len(events) - 1):
        a, b = offense[i], offense[i + 1]
        if a is not None and b is not None and a != b:
            groups.append((events[start:i + 1], a))
            start = i + 1
    groups.append((events[start:], offense[-1]))
    return groups


FOUL_LINK_WINDOW = 15   # events searched backward for the awarding foul
AND1_CLOCK_SLOP = 2     # seconds of clock drift tolerated on and-1s


def _link_foul(game_events, first_ft_gidx, clock, ft_team):
    """Nearest preceding foul by the opposing team, searched at the
    GAME level: the awarding foul often sits in the previous possession
    (rebound-scramble and dead-ball fouls), because fouls never change
    the running offense while FTs do. Same-clock match preferred."""
    same_clock, nearest = None, None
    lo = max(first_ft_gidx - FOUL_LINK_WINDOW, 0)
    period = game_events[first_ft_gidx]["period"]
    for j in range(first_ft_gidx - 1, lo - 1, -1):
        e = game_events[j]
        if e["period"] != period:
            break
        if e["etype"] == "foul" and e["team_id"] \
                and e["team_id"] != ft_team:
            if nearest is None:
                nearest = e
            if e["secs_remaining"] == clock:
                same_clock = e
                break
    return same_clock or nearest


def _find_and1_shot(game_events, first_ft_gidx, player, period, clock):
    """Made FG by the FT shooter at (nearly) the same clock, searched
    backward at the game level."""
    lo = max(first_ft_gidx - FOUL_LINK_WINDOW, 0)
    for j in range(first_ft_gidx - 1, lo - 1, -1):
        x = game_events[j]
        if x["period"] != period:
            break
        if (x["etype"] == "fg" and x["shot_made"]
                and x["player_id"] == player
                and x["secs_remaining"] is not None and clock is not None
                and abs(x["secs_remaining"] - clock) <= AND1_CLOCK_SLOP):
            return x
    return None


def attribute_possession(pevents, game_events):
    """-> dict of sfta/finisher/trip itemization + and1 shot events."""
    sfta = excluded = contamination = ft_and1 = ft_sf2 = ft_sf3 = 0
    ft_shooter = last_made = last_made_eid = None
    last_shooter = last_turnover = None
    and1_events = []

    trips = defaultdict(list)  # (player, period, clock) -> [events]
    for e in pevents:
        if e["etype"] == "fg":
            last_shooter = e["player_id"]
            if e["shot_made"]:
                last_made, last_made_eid = e["player_id"], e["event_id"]
        elif e["etype"] == "turnover" and e["player_id"]:
            last_turnover = e["player_id"]
        elif e["etype"] == "ft" and e["player_id"]:
            trips[(e["player_id"], e["period"], e["secs_remaining"])] \
                .append(e)

    for (player, period, clock), evs in sorted(
            trips.items(), key=lambda kv: kv[1][0]["_gidx"]):
        n = len(evs)
        gidx = evs[0]["_gidx"]
        foul = _link_foul(game_events, gidx, clock, evs[0]["team_id"])
        if foul is None or foul["foul_kind"] in EXCLUDED_TRIP_FOUL_KINDS \
                or foul["foul_kind"] == "offensive":
            excluded += n
            continue
        if n == 1:
            made = _find_and1_shot(game_events, gidx, player, period, clock)
            if made is None:
                contamination += 1  # 1-FT trip w/o basket: not attributable
                continue
            sfta += 1
            ft_and1 += 1
            ft_shooter = player
            and1_events.append((made["event_id"], player))
        elif n == 2:
            sfta += 2
            ft_sf2 += 2
            ft_shooter = player
        elif n == 3:
            sfta += 3
            ft_sf3 += 3
            ft_shooter = player
        else:
            excluded += n  # scorer glitch

    finisher = ft_shooter or last_made or last_turnover or last_shooter
    return {"sfta": sfta, "excluded_ft_count": excluded,
            "contamination_count": contamination, "ft_and1": ft_and1,
            "ft_sf2": ft_sf2, "ft_sf3": ft_sf3,
            "finisher_player_id": finisher,
            "and1_events": and1_events, "last_made_eid": last_made_eid}


def build_game(game_events, home_team_id):
    """One game's events (sorted) -> possession rows + shot-flag updates."""
    events = game_events
    for i, e in enumerate(events):
        e["_gidx"] = i
    offense = compute_offense(events)
    groups = split_possessions(events, offense)

    team_fouls = defaultdict(int)  # (team, min(period,4)) -> count
    teams = {e["team_id"] for e in events if e["team_id"]}

    rows, and1_rows, shot_flags = [], [], []
    prev_terminal_action, prev_offense = None, None

    for num, (pevents, off_team) in enumerate(groups, 1):
        # bonus state BEFORE this possession's events
        start_e = pevents[0]
        period_key = min(start_e["period"], 4)
        def_team = next((t for t in teams if t != off_team), None)
        in_bonus = 1 if team_fouls[(def_team, period_key)] >= 4 else 0

        # consume this possession's fouls into the running count
        for e in pevents:
            if e["etype"] == "foul" \
                    and e["foul_kind"] in COUNTED_FOUL_KINDS \
                    and e["team_id"]:
                team_fouls[(e["team_id"], min(e["period"], 4))] += 1

        att = attribute_possession(pevents, events)

        # terminal event: last FG, else last turnover, else last event
        terminal = next((e for e in reversed(pevents)
                         if e["etype"] == "fg"), None)
        t_action = None
        if terminal is not None:
            t_action = "made" if terminal["shot_made"] else "missed"
        else:
            terminal = next((e for e in reversed(pevents)
                             if e["etype"] == "turnover"), None)
            if terminal is not None:
                t_action = "turnover"
            else:
                terminal = pevents[-1]
                t_action = "other"

        # per-shot derived flags (used for ACB; EL keeps native flags)
        oreb_seen = False
        for e in pevents:
            if e["etype"] == "reb_off":
                oreb_seen = True
            elif e["etype"] == "fg":
                shot_flags.append({
                    "game_id": e["game_id"], "event_id": e["event_id"],
                    "second_chance": 1 if oreb_seen else 0,
                    "off_turnover": 1 if (prev_terminal_action == "turnover"
                                          and prev_offense != off_team)
                                    else 0})

        rows.append({
            "game_id": start_e["game_id"], "possession_number": num,
            "period": terminal["period"],
            "start_secs": start_e["secs_remaining"],
            "end_secs": terminal["secs_remaining"],
            "offense_team_id": off_team, "n_events": len(pevents),
            "in_bonus": in_bonus,
            "offense_is_home": 1 if off_team == home_team_id else 0,
            "terminal_event_id": terminal["event_id"],
            "terminal_action": t_action,
            "terminal_shot_value": terminal.get("shot_value"),
            "seconds_remaining_in_period": terminal["secs_remaining"],
            "score_margin": (terminal["score_home"]
                             - terminal["score_away"]),
            **{k: v for k, v in att.items()
               if k not in ("and1_events", "last_made_eid")},
        })
        for eid, pid in att["and1_events"]:
            and1_rows.append({"game_id": start_e["game_id"],
                              "event_id": eid, "player_id": pid,
                              "possession_number": num})
        prev_terminal_action, prev_offense = t_action, off_team
    return rows, and1_rows, shot_flags


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--league", choices=config.LEAGUES, default=None)
    args = ap.parse_args()
    leagues = [args.league] if args.league else config.LEAGUES

    con = sqlite3.connect(config.DB_PATH)
    for league in leagues:
        games = pd.read_sql(
            "SELECT game_id, season, home_team_id FROM games "
            "WHERE league = ?", con, params=(league,))
        events = pd.read_sql(
            "SELECT * FROM events WHERE league = ? "
            "ORDER BY game_id, event_id", con, params=(league,))
        print(f"[{league}] {len(games)} games, {len(events)} events")

        all_rows, all_and1, all_flags = [], [], []
        for g in games.itertuples():
            ge = events[events.game_id == g.game_id]
            if ge.empty:
                continue
            rows, and1_rows, flags = build_game(
                ge.to_dict("records"), g.home_team_id)
            for r in rows:
                r["league"], r["season"] = league, g.season
            all_rows.extend(rows)
            all_and1.extend(and1_rows)
            all_flags.extend(flags)

        pdf = pd.DataFrame(all_rows)
        adf = pd.DataFrame(all_and1, columns=["game_id", "event_id",
                                              "player_id",
                                              "possession_number"])
        adf["league"] = league
        # possession-derived shot context flags for BOTH leagues; these
        # never encode make/miss (unlike the EL Points-feed native flags)
        fdf = pd.DataFrame(all_flags, columns=["game_id", "event_id",
                                               "second_chance",
                                               "off_turnover"])
        fdf["league"] = league
        for name, df in (("possessions", pdf), ("and1_shots", adf),
                         ("possession_shot_flags", fdf)):
            con.execute(f"DELETE FROM {name} WHERE league = ?", (league,)) \
                if _table_exists(con, name) else None
            df.to_sql(name, con, if_exists="append", index=False)

        n_poss = len(pdf)
        per_game = n_poss / max(len(games), 1)
        print(f"  possessions: {n_poss} ({per_game:.0f}/game), "
              f"sfta sum {pdf.sfta.sum()}, and1 {len(adf)}, "
              f"excluded {pdf.excluded_ft_count.sum()}, "
              f"contaminated {pdf.contamination_count.sum()}")
    con.execute("CREATE INDEX IF NOT EXISTS ix_psf "
                "ON possession_shot_flags(league, game_id, event_id)")
    con.commit()
    con.close()


def _table_exists(con, name):
    return con.execute(
        "SELECT 1 FROM sqlite_master WHERE type='table' AND name=?",
        (name,)).fetchone() is not None


if __name__ == "__main__":
    main()
