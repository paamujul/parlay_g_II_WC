"""Backtest: calibrate and validate the team-score -> match-outcome model.

Model (ordered logistic on an Elo-like scale):
    strength(team)  = alpha + beta * team_score        (our player-sum teams)
    strength(opp)   = opponent Elo rating              (eloratings.net-style)
    d               = strength(team) - strength(opp)
    P(win)          = sigmoid((d - c) / s)
    P(win or draw)  = sigmoid((d + c) / s)
    P(draw)         = P(win or draw) - P(win)
with s = 400/ln(10) (standard Elo logistic scale) and c > 0 a draw-width
parameter. alpha, beta, c are fit by maximum likelihood on 2023-2026 matches.

Validation: time split - train on matches before 2025-07-01, test on
everything after (incl. the whole 2026 World Cup run). Metrics: accuracy,
3-class Brier score, log loss, vs. an "always predict win" baseline.

Honest limitations (also in README): results are 90-minute results; only two
teams carry player-sum scores, so beta is identified mainly by the Spain vs
Argentina quality gap; team score is held constant across the window.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from scoring import score_teams

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "output"
S = 400.0 / np.log(10.0)  # Elo logistic scale
RESULT_CODE = {"W": 2, "D": 1, "L": 0}


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def probs(d, c):
    """Return array [P(loss), P(draw), P(win)] for strength diff d."""
    p_win = sigmoid((d - c) / S)
    p_wd = sigmoid((d + c) / S)
    return np.stack([1.0 - p_wd, p_wd - p_win, p_win], axis=-1)


def load_matches(team_scores):
    df = pd.read_csv(ROOT / "data" / "matches.csv", parse_dates=["date"])
    df["team_score"] = df["team"].map(team_scores)
    df["y"] = df["result90"].map(RESULT_CODE)
    return df


def nll(params, team_score, opp_elo, y):
    alpha, beta, c = params
    d = alpha + beta * team_score - opp_elo
    p = probs(d, max(c, 1e-6))
    return -np.log(np.clip(p[np.arange(len(y)), y], 1e-12, 1)).sum()


def fit(df):
    args = (df["team_score"].values, df["opp_elo"].values, df["y"].values)
    x0 = np.array([1400.0, 1.0, 80.0])
    res = minimize(nll, x0, args=args, method="Nelder-Mead",
                   options={"maxiter": 20000, "xatol": 1e-6, "fatol": 1e-8})
    return res.x


def evaluate(df, params, label):
    alpha, beta, c = params
    d = alpha + beta * df["team_score"].values - df["opp_elo"].values
    p = probs(d, c)
    y = df["y"].values
    onehot = np.eye(3)[y]
    acc = float((p.argmax(axis=1) == y).mean())
    brier = float(((p - onehot) ** 2).sum(axis=1).mean())
    logloss = float(-np.log(np.clip(p[np.arange(len(y)), y], 1e-12, 1)).mean())
    base_acc = float((y == 2).mean())  # baseline: always predict a win
    print(f"{label:<28} n={len(df):3d}  acc={acc:.3f} (baseline always-W {base_acc:.3f})  "
          f"brier={brier:.3f}  logloss={logloss:.3f}")
    return {"n": len(df), "accuracy": acc, "baseline_accuracy": base_acc,
            "brier": brier, "logloss": logloss}


def main():
    OUT.mkdir(exist_ok=True)
    scores = {t: b["team_score"] for t, b in score_teams().items()}
    df = load_matches(scores)

    split = pd.Timestamp("2025-07-01")
    train, test = df[df["date"] < split], df[df["date"] >= split]

    params_train = fit(train)
    print("Backtest (time split at 2025-07-01)")
    m_train = evaluate(train, params_train, "  train (2023-01..2025-06)")
    m_test = evaluate(test, params_train, "  test  (2025-07..2026-07)")

    # Final calibration on all data for the live prediction step.
    params_all = fit(df)
    m_all = evaluate(df, params_all, "  refit on all matches")
    alpha, beta, c = params_all
    for team in ("Spain", "Argentina"):
        print(f"  {team}: team_score={scores[team]:.1f} -> Elo-equivalent "
              f"{alpha + beta * scores[team]:.0f}")

    (OUT / "model_params.json").write_text(json.dumps({
        "alpha": alpha, "beta": beta, "draw_width_c": c, "elo_scale_s": S,
        "team_scores": scores,
        "metrics": {"train": m_train, "test": m_test, "all": m_all},
    }, indent=2))
    print(f"\nSaved calibrated parameters to {OUT / 'model_params.json'}")


if __name__ == "__main__":
    main()
