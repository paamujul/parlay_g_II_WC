"""Live in-game win-probability ML model (the end goal).

Trains a gradient-boosted classifier that maps a live match state to
P(home win / draw / away win) at 90 minutes. Because we have no proprietary
event feed, training data comes from a minute-by-minute Poisson match
simulator driven by the same backtest-calibrated strength model - so the
pipeline runs end to end today, and the simulator rows can be swapped 1:1
for real event-feed rows (API-Football, StatsBomb, Opta) later.

Features per snapshot:
  minute, score_diff, elo_diff, red_card_diff, lambda_remaining_diff

Outputs:
  - holdout log-loss / accuracy vs a naive baseline
  - live win-probability trajectory for a sample Spain-Argentina scenario
  - saved model at output/live_model.joblib
"""

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import accuracy_score, log_loss
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "output"
RNG = np.random.default_rng(42)

TOTAL_GOALS_90 = 2.35
GOALS_PER_ELO = 1.0 / 200.0
RED_CARD_RATE_PER_MIN = 0.0004      # ~3.5% of matches see a red
RED_CARD_LAMBDA_PENALTY = 0.35      # scoring-rate multiplier hit when a man down

FEATURES = ["minute", "score_diff", "elo_diff", "red_card_diff", "lam_diff_remaining"]


def simulate_matches(n_matches: int, elo_diff_range=(-400, 400), snapshots_per_match=6):
    """Simulate matches minute-by-minute; emit (state, final-outcome) rows."""
    rows = []
    for _ in range(n_matches):
        elo_diff = RNG.uniform(*elo_diff_range)
        mu = elo_diff * GOALS_PER_ELO
        lam_h = max(TOTAL_GOALS_90 / 2 + mu / 2, 0.2) / 90.0
        lam_a = max(TOTAL_GOALS_90 / 2 - mu / 2, 0.2) / 90.0
        goals_h = goals_a = red_h = red_a = 0
        snap_minutes = set(RNG.integers(1, 90, snapshots_per_match).tolist())
        match_rows = []
        for minute in range(1, 91):
            mult_h = RED_CARD_LAMBDA_PENALTY if red_h > red_a else 1.0
            mult_a = RED_CARD_LAMBDA_PENALTY if red_a > red_h else 1.0
            if minute in snap_minutes:
                lam_h_now, lam_a_now = lam_h * mult_h, lam_a * mult_a
                match_rows.append({
                    "minute": minute,
                    "score_diff": goals_h - goals_a,
                    "elo_diff": elo_diff,
                    "red_card_diff": red_h - red_a,
                    "lam_diff_remaining": (lam_h_now - lam_a_now) * (90 - minute),
                })
            if RNG.random() < lam_h * mult_h:
                goals_h += 1
            if RNG.random() < lam_a * mult_a:
                goals_a += 1
            if RNG.random() < RED_CARD_RATE_PER_MIN:
                if RNG.random() < 0.5:
                    red_h += 1
                else:
                    red_a += 1
        outcome = 2 if goals_h > goals_a else (1 if goals_h == goals_a else 0)
        for r in match_rows:
            r["outcome"] = outcome
        rows.extend(match_rows)
    return pd.DataFrame(rows)


def train():
    print("Simulating 40,000 training matches ...")
    df = simulate_matches(40_000)
    X, y = df[FEATURES], df["outcome"]
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=0)

    model = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.08,
                                           max_depth=5, random_state=0)
    model.fit(X_tr, y_tr)

    p = model.predict_proba(X_te)
    print(f"Holdout: log-loss={log_loss(y_te, p):.4f}  "
          f"accuracy={accuracy_score(y_te, p.argmax(axis=1)):.3f}  "
          f"(naive always-most-common={y_te.value_counts(normalize=True).max():.3f})")
    joblib.dump(model, OUT / "live_model.joblib")
    return model


def live_prob(model, minute, score_diff, elo_diff, red_card_diff=0):
    mu = elo_diff * GOALS_PER_ELO
    lam_h = max(TOTAL_GOALS_90 / 2 + mu / 2, 0.2) / 90.0
    lam_a = max(TOTAL_GOALS_90 / 2 - mu / 2, 0.2) / 90.0
    # red_card_diff = home_reds - away_reds; a positive value penalises home.
    mult_h = RED_CARD_LAMBDA_PENALTY if red_card_diff > 0 else 1.0
    mult_a = RED_CARD_LAMBDA_PENALTY if red_card_diff < 0 else 1.0
    x = pd.DataFrame([{
        "minute": minute, "score_diff": score_diff, "elo_diff": elo_diff,
        "red_card_diff": red_card_diff,
        "lam_diff_remaining": (lam_h * mult_h - lam_a * mult_a) * (90 - minute),
    }])
    return model.predict_proba(x)[0]  # [away win, draw, home win]


def demo(model):
    params = json.loads((OUT / "model_params.json").read_text())
    scores = params["team_scores"]
    elo_diff = (params["alpha"] + params["beta"] * scores["Spain"]) - \
               (params["alpha"] + params["beta"] * scores["Argentina"])

    scenario = [
        (1, 0, 0, "Kickoff"),
        (25, 0, 0, "Still 0-0"),
        (40, 1, 0, "GOAL Spain (say, Yamal cuts inside)"),
        (46, 1, 0, "Half time 1-0"),
        (65, 1, 0, "Hour mark, Spain lead"),
        (72, 0, 0, "GOAL Argentina - Messi equalises, 1-1"),
        (85, 0, 0, "1-1, five to play"),
        (88, 1, 0, "GOAL Spain, 2-1!"),
    ]
    print(f"\nLive win-probability demo: Spain (home slot) vs Argentina, elo_diff={elo_diff:+.0f}")
    print(f"{'min':>4} {'Spain':>7} {'Draw':>7} {'Argentina':>10}   event")
    for minute, sd, rd, event in scenario:
        p_a, p_d, p_h = live_prob(model, minute, sd, elo_diff, rd)
        print(f"{minute:>4} {p_h:>7.1%} {p_d:>7.1%} {p_a:>10.1%}   {event}")


if __name__ == "__main__":
    OUT.mkdir(exist_ok=True)
    model = train()
    demo(model)
