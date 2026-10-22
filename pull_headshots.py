#!/usr/bin/env python3
"""Player headshot URL maps, one JSON per league.

Why a map instead of a URL template: only the NBA, the WNBA and the ABA Liga
publish a headshot at a URL derivable from the player id we already store.
EuroLeague and EuroCup key theirs on a media UUID, Liga ACB on a hashed media
path, the Greek Basket League on a Genius asset hash, and Lega Basket Serie A on
a 24-character asset key. None of those can be computed from a player id, so the
id -> URL join is resolved once here and shipped.

Writes site/public/headshots-{LG}.json:

    {"meta": {"league", "source", "generated", "players", "matched", "coverage"},
     "urls": {"<playerId>": "<url>"}}

A NEW artifact on purpose: data-{LG}.json keeps its exact shape, and the front
end fetches this lazily, only on the routes that show a face.

Four of the six leagues need no network call at all. ACB, GBL and LBA photo URLs
are already inside the pull cache, sitting next to the players; ABA's are
arithmetic. Only EuroLeague and EuroCup are fetched.

Not included: the Basketball Bundesliga. Its
/images/player/portrait/{season}/{id} endpoint answers 200 for any id with an
identical 87,072-byte placeholder SVG (verified by md5 across two players,
including one whose own feed record advertises an imageUrl), so there is nothing
to map.

Run:  python3 pull_headshots.py [LG ...]      (default: every supported league)
"""
import json
import re
import sys
import time
import unicodedata
import urllib.parse
import urllib.request
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import config

HERE = Path(__file__).resolve().parent
CACHE = HERE / "cache"
OUT = HERE / "site" / "public"

# The same slug the Genius adapter derives its player ids from. Imported rather
# than reimplemented so the two cannot drift: a divergence here would silently
# drop every GBL face.
sys.path.insert(0, str(HERE))
from leagues.genius import _slug as genius_slug  # noqa: E402

UA = {"User-Agent": config.EL_UA}

# Per-league request parameters that pick a sensibly sized variant. EuroLeague's
# originals are 750x1000 / ~549 KB, which would make the headshot a bad LCP
# element; the CDN honours format+width and returns ~12 KB instead. These
# parameters are undocumented, so `--check` verifies the bytes.
EL_VARIANT = "?format=webp&w=200"
LBA_VARIANT = "thumb"          # 'large' is ~137 KB; 'small'/'medium' are 403


def qualified(lg):
    """Every player with a page: the ones carrying a percentile."""
    d = json.loads((OUT / f"data-{lg}.json").read_text())
    latest = {}
    for season in d["meta"]["seasons"]:
        for r in d["leaderboard"]:
            if r["season"] == season and r.get("pct") is not None:
                latest[str(r["id"])] = season
    return latest


def _get_json(url):
    req = urllib.request.Request(url, headers=UA)
    with urllib.request.urlopen(req, timeout=config.TIMEOUT) as r:
        return json.load(r)


def euroleague(lg, players):
    """EuroLeague / EuroCup: the competition's own people feed, one page at a
    time. A person's headshot does not change between seasons, so every season
    is unioned to catch players whose photo only appears in an older roster."""
    comp = "E" if lg == "EL" else "U"
    urls = {}
    for season in sorted({s for s in qualified(lg).values()} | set(json.loads(
            (OUT / f"data-{lg}.json").read_text())["meta"]["seasons"])):
        code = f"{comp}{season[:4]}"
        offset = 0
        while True:
            feed = (f"{config.EL_API}/v2/competitions/{comp}/seasons/{code}"
                    f"/people?offset={offset}&limit=500")
            try:
                page = _get_json(feed)
            except Exception as e:
                print(f"    {code} offset {offset}: {e}")
                break
            rows = page.get("data") or []
            for r in rows:
                if r.get("typeName") != "Player":
                    continue
                shot = (r.get("images") or {}).get("headshot")
                if shot:
                    urls.setdefault(f"P{r['person']['code']}", []).append(
                        shot + EL_VARIANT)
            if len(rows) < 500:
                break
            offset += 500
            time.sleep(config.EL_SLEEP)
        time.sleep(config.EL_SLEEP)
    return urls, f"{config.EL_API}/v2/competitions/{comp}/.../people"


def acb(lg, players):
    """Liga ACB: already in the cached boxscores. The boxscore's player.id is
    the id we export, so this is a straight read with no join to get wrong."""
    urls = {}
    for f in sorted((CACHE / "ACB" / "box").glob("*.json")):
        try:
            d = json.loads(f.read_text())
        except Exception:
            continue
        for tb in d.get("teamBoxscores", []):
            for sp in tb.get("statsByPeriods", []):
                for pr in sp.get("stats", {}).get("players", []):
                    p = pr.get("player") or {}
                    pid, shot = p.get("id"), p.get("headshotImageNoBackgroundUrl")
                    if pid and shot:
                        urls.setdefault(str(pid), []).append(shot)
    return urls, "cached ACB boxscores (player.headshotImageNoBackgroundUrl)"


def gbl(lg, players):
    """Greek Basket League: already in the cached Genius game files. The feed
    carries no player id, so the join is the adapter's own name slug — the one
    weak link in this whole step, which is why the match rate is reported and
    the caller fails the pull if it collapses."""
    urls = {}
    for f in sorted((CACHE / "GBL" / "data").glob("*.json")):
        try:
            d = json.loads(f.read_text())
        except Exception:
            continue
        for tno in ("1", "2"):
            for p in (d.get("tm", {}).get(tno, {}).get("pl", {}) or {}).values():
                pid = genius_slug(p.get("internationalFamilyName"),
                                  p.get("internationalFirstName"))
                for shot in (p.get("photoS"), p.get("photoT")):
                    if pid and shot:
                        urls.setdefault(pid, []).append(
                            shot.replace("http://", "https://", 1))
    return urls, "cached GBL Genius game files (tm.*.pl.*.photoS)"


def lba(lg, players):
    """Lega Basket Serie A: the cached boxscores carry a 24-character media key
    per player row; the public bucket serves variants under it."""
    urls = {}
    for f in sorted((CACHE / "LBA" / "match").glob("*.json")):
        try:
            d = json.loads(f.read_text())
        except Exception:
            continue
        for side in ("ht", "vt"):
            for r in d.get("scores", {}).get(side, {}).get("rows", []):
                key, pid = r.get("player_p_key"), r.get("player_id")
                if key and pid:
                    urls.setdefault(str(pid), []).append(
                        "https://lba-media.s3.eu-south-1.amazonaws.com"
                        f"/variants/{key}/{LBA_VARIANT}"
                    )
    return urls, "cached LBA boxscores (scores.*.rows[].player_p_key)"


def aba(lg, players):
    """ABA Liga: URL-addressable from the id and the season, so this needs no
    network and no cache — it is arithmetic. Kept in the same map format as the
    others so the front end has one code path rather than two."""
    urls = {
        pid: ["https://www.aba-liga.com/stats/img/foto/"
              f"sez{int(season[:4]) - 2000}/{pid}.png"]
        for pid, season in players.items()
    }
    return urls, "aba-liga.com/stats/img/foto/sez{YY}/{id}.png (derived)"


RESOLVERS = {
    "EL": euroleague,
    "EUC": euroleague,
    "ACB": acb,
    "GBL": gbl,
    "LBA": lba,
    "ABA": aba,
}

# Below this share of players matched, something upstream changed shape and the
# map should not ship half-empty and unnoticed.
MIN_COVERAGE = 0.5

_host_ok = {}


def host_alive(url):
    """One probe per distinct host, cached. A host that fails takes every URL on
    it down with it, so per-URL checks would be 2,000 requests to learn the same
    eight facts. Genius serves the Greek feed from three hosts and one of them
    (se-img.dcd-production) no longer resolves at all, which is exactly the case
    this catches — 127 of those players have the same photo on a live host."""
    host = urllib.parse.urlparse(url).netloc
    if host not in _host_ok:
        try:
            req = urllib.request.Request(url, headers=UA, method="GET")
            with urllib.request.urlopen(req, timeout=20) as r:
                _host_ok[host] = r.status == 200 and (
                    r.headers.get("content-type", "").startswith("image/"))
        except Exception:
            _host_ok[host] = False
        print(f"      host {'ok  ' if _host_ok[host] else 'DEAD'}  {host}")
    return _host_ok[host]


def pick(candidates):
    """First candidate on a host that answers with an image."""
    for url in candidates:
        if host_alive(url):
            return url
    return None


def run(lg):
    players = qualified(lg)
    candidates, source = RESOLVERS[lg](lg, players)
    kept = {}
    for pid in players:
        chosen = pick(candidates.get(pid, []))
        if chosen:
            kept[pid] = chosen
    coverage = len(kept) / len(players) if players else 0.0
    payload = {
        "meta": {
            "league": lg,
            "source": source,
            "generated": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "players": len(players),
            "matched": len(kept),
            "coverage": round(coverage, 4),
        },
        "urls": dict(sorted(kept.items())),
    }
    status = "ok " if coverage >= MIN_COVERAGE else "LOW"
    print(f"  [{status}] {lg:5} {len(kept):5}/{len(players):<5} "
          f"({coverage:6.1%})  {source}")
    if coverage < MIN_COVERAGE:
        return None
    (OUT / f"headshots-{lg}.json").write_text(json.dumps(payload) + "\n")
    return payload


def main(argv):
    leagues = [a.upper() for a in argv[1:]] or list(RESOLVERS)
    bad = [lg for lg in leagues if lg not in RESOLVERS]
    if bad:
        sys.exit(f"no headshot source for {bad}; supported: {list(RESOLVERS)}")
    print("=== headshot maps ===")
    failed = [lg for lg in leagues if run(lg) is None]
    if failed:
        sys.exit(f"coverage under {MIN_COVERAGE:.0%} for {failed}; not written")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
