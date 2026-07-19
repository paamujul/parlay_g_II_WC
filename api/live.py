"""Vercel serverless function: live Spain-Argentina win probability + player prop odds.

GET /api/live
  -> fetches the ESPN public match feed (no API key), reads score, clock,
     red cards, shots-on-target and key events, then returns:
       - live P(Spain win / draw / Argentina win) at full time
       - fair player prop odds (anytime scorer, remaining-time) per starter
       - match total-goals over/under probabilities
       - the latest key events

Win-probability math is the same Poisson goal model calibrated in
src/backtest.py (Spain 1.30 xG/90, Argentina 1.05 xG/90 on neutral turf),
evaluated analytically over the minutes remaining, with adjustments for:
  - current scoreline (dominant factor)
  - red cards (a team a man down scores at 0.35x rate)
  - live momentum via shots-on-target share (mild, bounded effect)

Stdlib only - deploys on Vercel's Python runtime with zero dependencies.
"""

import json
import math
import urllib.request
from http.server import BaseHTTPRequestHandler

EVENT_URL = ("https://site.api.espn.com/apis/site/v2/sports/soccer/"
             "fifa.world/summary?event=760517")

# Calibrated in src/backtest.py -> output/final_prediction.json
LAMBDA_90 = {"Spain": 1.30, "Argentina": 1.05}
RED_CARD_MULT = 0.35
MAX_GOALS = 12  # truncation for Poisson sums

# Attacking share of each team's goals, derived from player scores in
# output/scores.json (per-90 goal contribution weights, normalised).
ATTACK_SHARES = {
    "Spain": {
        "Lamine Yamal": 0.27, "Mikel Oyarzabal": 0.20, "Dani Olmo": 0.14,
        "Alex Baena": 0.11, "Pedro Porro": 0.07, "Fabian Ruiz": 0.06,
        "Rodri": 0.05, "Marc Cucurella": 0.04, "Others": 0.06,
    },
    "Argentina": {
        "Julian Alvarez": 0.27, "Lionel Messi": 0.26, "Enzo Fernandez": 0.12,
        "Alexis Mac Allister": 0.10, "Rodrigo De Paul": 0.07,
        "Cristian Romero": 0.05, "Leandro Paredes": 0.04, "Others": 0.09,
    },
}


def poisson_pmf(lam, k):
    return math.exp(-lam) * lam ** k / math.factorial(k)


def outcome_probs(score_h, score_a, lam_h, lam_a):
    """P(home win / draw / away win) at FT given remaining-time lambdas."""
    p_h = p_d = p_a = 0.0
    for i in range(MAX_GOALS + 1):
        for j in range(MAX_GOALS + 1):
            p = poisson_pmf(lam_h, i) * poisson_pmf(lam_a, j)
            total_h, total_a = score_h + i, score_a + j
            if total_h > total_a:
                p_h += p
            elif total_h == total_a:
                p_d += p
            else:
                p_a += p
    return p_h, p_d, p_a


def to_odds(p):
    """Fair (no-vig) decimal + American odds for probability p."""
    p = min(max(p, 1e-4), 0.9999)
    decimal = 1.0 / p
    american = round(-100 * p / (1 - p)) if p >= 0.5 else round(100 * (1 - p) / p)
    return {"prob": round(p, 4), "decimal": round(decimal, 2),
            "american": f"{american:+d}"}


def parse_minute(status):
    clock = status.get("displayClock", "0'")
    digits = "".join(ch for ch in clock.split("+")[0] if ch.isdigit())
    return int(digits) if digits else 0, clock


def fetch_espn():
    req = urllib.request.Request(EVENT_URL, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=15) as r:
        return json.load(r)


def team_stat(boxscore, side_index, *names):
    """Pull a named stat (e.g. shots on target) for one team, else None."""
    try:
        stats = boxscore["teams"][side_index]["statistics"]
        for s in stats:
            if s.get("name") in names:
                return float(str(s.get("displayValue", "0")).rstrip("%"))
    except (KeyError, IndexError, ValueError):
        pass
    return None


def build_payload():
    data = fetch_espn()
    comp = data["header"]["competitions"][0]
    status = comp["status"]
    state = status["type"]["state"]  # pre | in | post

    teams = {c["homeAway"]: c for c in comp["competitors"]}
    home_name = teams["home"]["team"]["displayName"]      # Spain
    away_name = teams["away"]["team"]["displayName"]      # Argentina
    score_h = int(teams["home"].get("score") or 0)
    score_a = int(teams["away"].get("score") or 0)

    minute, clock = parse_minute(status)
    if state == "pre":
        minute, clock = 0, "0'"
    remaining = max(90 - minute, 0) if state != "post" else 0

    # Key events: red cards + recent feed
    events, reds = [], {home_name: 0, away_name: 0}
    for e in data.get("keyEvents", []):
        etype = e.get("type", {}).get("text", "")
        eteam = e.get("team", {}).get("displayName", "")
        players = [p.get("athlete", {}).get("displayName", "")
                   for p in e.get("participants", [])]
        if "Red Card" in etype and eteam in reds:
            reds[eteam] += 1
        events.append({"minute": e.get("clock", {}).get("displayValue", ""),
                       "type": etype, "team": eteam, "players": players})

    # Momentum from shots-on-target share (bounded 0.85x..1.15x)
    sot_h = team_stat(data.get("boxscore", {}), 0, "shotsOnTarget")
    sot_a = team_stat(data.get("boxscore", {}), 1, "shotsOnTarget")
    mom_h = mom_a = 1.0
    if state == "in" and sot_h is not None and sot_a is not None and sot_h + sot_a > 0:
        share = sot_h / (sot_h + sot_a)
        mom_h = 0.85 + 0.30 * share
        mom_a = 0.85 + 0.30 * (1 - share)

    lam_h = (LAMBDA_90[home_name] / 90.0) * remaining * mom_h * \
        (RED_CARD_MULT if reds[home_name] > reds[away_name] else 1.0)
    lam_a = (LAMBDA_90[away_name] / 90.0) * remaining * mom_a * \
        (RED_CARD_MULT if reds[away_name] > reds[home_name] else 1.0)

    p_h, p_d, p_a = outcome_probs(score_h, score_a, lam_h, lam_a)

    # Player props: P(scores in the remaining minutes)
    props = []
    for team, lam_team in ((home_name, lam_h), (away_name, lam_a)):
        for player, share in ATTACK_SHARES[team].items():
            if player == "Others":
                continue
            props.append({"player": player, "team": team,
                          **to_odds(1 - math.exp(-lam_team * share))})
    props.sort(key=lambda r: -r["prob"])

    # Match totals (goals so far + Poisson remaining)
    lam_total = lam_h + lam_a
    goals_now = score_h + score_a
    totals = {}
    for line in (1.5, 2.5, 3.5):
        need = max(int(math.floor(line + 1)) - goals_now, 0)
        p_over = 1 - sum(poisson_pmf(lam_total, k) for k in range(need))
        totals[f"over_{line}"] = to_odds(p_over)
        totals[f"under_{line}"] = to_odds(1 - p_over)

    return {
        "match": f"{home_name} vs {away_name} - WC 2026 Final",
        "state": state, "clock": clock, "minute": minute,
        "score": {home_name: score_h, away_name: score_a},
        "red_cards": reds,
        "win_probability": {home_name: round(p_h, 4), "Draw": round(p_d, 4),
                            away_name: round(p_a, 4)},
        "note_if_draw": "Draw at FT goes to extra time and penalties.",
        "player_props_anytime_scorer": props,
        "total_goals": totals,
        "key_events": events[-10:],
        "model": {"lambda_remaining": {home_name: round(lam_h, 3),
                                       away_name: round(lam_a, 3)},
                  "momentum_mult": {home_name: round(mom_h, 3),
                                    away_name: round(mom_a, 3)}},
    }


class handler(BaseHTTPRequestHandler):
    def do_GET(self):
        try:
            body = json.dumps(build_payload()).encode()
            code = 200
        except Exception as exc:  # surface feed hiccups to the client politely
            body = json.dumps({"error": str(exc)}).encode()
            code = 502
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)
