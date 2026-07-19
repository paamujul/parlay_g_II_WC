"""Predict the WC 2026 final: Spain vs Argentina (neutral venue, MetLife).

Uses the backtest-calibrated strength model to get expected goals for each
side, then Monte-Carlo simulates the match (Poisson goals, 90 min + extra
time + penalties) to produce:
  - 90-minute win/draw/loss probabilities
  - most likely scorelines
  - probability of lifting the trophy
"""

import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "output"

TOTAL_GOALS_90 = 2.35   # expected total goals in an elite international final
GOALS_PER_ELO = 1.0 / 200.0  # ~100 Elo points ~ 0.5 goal expected margin
N_SIMS = 200_000
RNG = np.random.default_rng(7)


def main():
    params = json.loads((OUT / "model_params.json").read_text())
    scores = params["team_scores"]
    alpha, beta = params["alpha"], params["beta"]

    elo = {t: alpha + beta * s for t, s in scores.items()}
    d = elo["Spain"] - elo["Argentina"]
    mu = d * GOALS_PER_ELO  # expected goal margin, Spain minus Argentina

    lam_esp = max(TOTAL_GOALS_90 / 2 + mu / 2, 0.25)
    lam_arg = max(TOTAL_GOALS_90 / 2 - mu / 2, 0.25)

    g_esp = RNG.poisson(lam_esp, N_SIMS)
    g_arg = RNG.poisson(lam_arg, N_SIMS)

    esp_w90 = (g_esp > g_arg).mean()
    draw90 = (g_esp == g_arg).mean()
    arg_w90 = (g_esp < g_arg).mean()

    # Extra time for the draws (30 min => lambda/3), then penalties ~55/45
    # to the higher-strength side (shootout is mostly a coin flip).
    tied = g_esp == g_arg
    n_tied = int(tied.sum())
    et_esp = RNG.poisson(lam_esp / 3.0, n_tied)
    et_arg = RNG.poisson(lam_arg / 3.0, n_tied)
    pen_edge = 0.5 + np.clip(d / 2000.0, -0.05, 0.05)
    pens = RNG.random(n_tied) < pen_edge

    esp_trophy = esp_w90 + (tied.mean() * ((et_esp > et_arg).mean()
                 + ((et_esp == et_arg) & pens).mean()))
    arg_trophy = 1.0 - esp_trophy

    print("=== WC 2026 FINAL: Spain vs Argentina (MetLife, 2026-07-19) ===\n")
    print(f"Team scores: Spain {scores['Spain']:.1f}  |  Argentina {scores['Argentina']:.1f}")
    print(f"Elo-equivalent: Spain {elo['Spain']:.0f}  |  Argentina {elo['Argentina']:.0f}"
          f"  (diff {d:+.0f})")
    print(f"Expected goals: Spain {lam_esp:.2f} - {lam_arg:.2f} Argentina\n")
    print(f"90-minute result:  Spain {esp_w90:.1%}  |  Draw {draw90:.1%}  |  Argentina {arg_w90:.1%}")
    print(f"Wins the trophy:   Spain {esp_trophy:.1%}  |  Argentina {arg_trophy:.1%}\n")

    print("Most likely scorelines (90 min):")
    lines = {}
    for a, b in zip(g_esp, g_arg):
        lines[(a, b)] = lines.get((a, b), 0) + 1
    for (a, b), n in sorted(lines.items(), key=lambda kv: -kv[1])[:6]:
        print(f"  Spain {a}-{b} Argentina : {n / N_SIMS:.1%}")

    (OUT / "final_prediction.json").write_text(json.dumps({
        "match": "Spain vs Argentina, WC 2026 Final",
        "team_scores": scores,
        "elo_equivalent": {k: round(v, 1) for k, v in elo.items()},
        "expected_goals": {"Spain": round(lam_esp, 2), "Argentina": round(lam_arg, 2)},
        "p90": {"Spain": round(float(esp_w90), 4), "Draw": round(float(draw90), 4),
                "Argentina": round(float(arg_w90), 4)},
        "trophy": {"Spain": round(float(esp_trophy), 4),
                   "Argentina": round(float(arg_trophy), 4)},
    }, indent=2))


if __name__ == "__main__":
    main()
