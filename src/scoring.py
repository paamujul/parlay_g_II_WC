"""Player and team scoring from club stats (2023-24 to 2025-26).

Each player-season gets a 0-100 score from three signals:
  - availability : minutes played vs a full workload (~3400 min all comps)
  - performance  : goal contributions per 90 vs a positional baseline
                   (goalkeepers use clean-sheet rate instead)
  - league level : multiplier for the strength of the league played in

Player score = recency-weighted average of season scores.
Team score   = sum of the 11 starters' scores (per project spec).
"""

import json
from pathlib import Path

DATA = Path(__file__).resolve().parent.parent / "data" / "players.json"
OUT = Path(__file__).resolve().parent.parent / "output"

LEAGUE_STRENGTH = {
    "Premier League": 1.00,
    "La Liga": 0.98,
    "Serie A": 0.93,
    "Bundesliga": 0.93,
    "Ligue 1": 0.90,
    "Saudi Pro League": 0.62,
    "MLS": 0.60,
    "Argentine Primera": 0.66,
}

# Expected (goals + 0.75*assists) per 90 for a solid starter at each position.
POS_BASELINE = {"CB": 0.07, "FB": 0.18, "DM": 0.20, "CM": 0.32, "AM": 0.50, "W": 0.55, "FW": 0.65}

SEASON_WEIGHTS = {"2023-24": 0.20, "2024-25": 0.35, "2025-26": 0.45}

FULL_SEASON_MINUTES = 3400
GK_CS_BASELINE = 0.42          # clean sheets per app for an elite GK
PERF_CAP = 1.8                 # cap on performance ratio so one outlier season can't dominate


def season_score(pos: str, s: dict) -> float:
    availability = min(s["minutes"] / FULL_SEASON_MINUTES, 1.0)
    league = LEAGUE_STRENGTH[s["league"]]
    if pos == "GK":
        cs_rate = s.get("clean_sheets", 0) / max(s["apps"], 1)
        perf = min(cs_rate / GK_CS_BASELINE, 1.5) / 1.5
    else:
        per90 = (s["goals"] + 0.75 * s["assists"]) / max(s["minutes"] / 90.0, 1.0)
        perf = min(per90 / POS_BASELINE[pos], PERF_CAP) / PERF_CAP
    return 100.0 * league * (0.45 * availability + 0.55 * perf)


def player_score(player: dict) -> dict:
    rows, total, wsum = [], 0.0, 0.0
    for s in player["seasons"]:
        sc = season_score(player["pos"], s)
        w = SEASON_WEIGHTS[s["season"]]
        total += w * sc
        wsum += w
        rows.append({"season": s["season"], "club": s["club"], "score": round(sc, 1)})
    return {
        "name": player["name"],
        "pos": player["pos"],
        "score": round(total / wsum, 1),
        "seasons": rows,
    }


def score_teams() -> dict:
    data = json.loads(DATA.read_text())
    out = {}
    for team, players in data["teams"].items():
        scored = sorted((player_score(p) for p in players), key=lambda r: -r["score"])
        out[team] = {"players": scored, "team_score": round(sum(r["score"] for r in scored), 1)}
    return out


def main():
    OUT.mkdir(exist_ok=True)
    result = score_teams()
    (OUT / "scores.json").write_text(json.dumps(result, indent=2))
    for team, block in result.items():
        print(f"\n=== {team}  |  TEAM SCORE: {block['team_score']} ===")
        for r in block["players"]:
            print(f"  {r['score']:5.1f}  {r['pos']:<3} {r['name']}")
    diff = result["Spain"]["team_score"] - result["Argentina"]["team_score"]
    print(f"\nTeam score difference (Spain - Argentina): {diff:+.1f}")


if __name__ == "__main__":
    main()
