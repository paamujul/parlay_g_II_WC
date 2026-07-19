# Parlay-G WC-2026 — Spain vs Argentina Final Predictor

Player-level performance model for the 2026 World Cup final (Spain vs
Argentina, MetLife Stadium, 2026-07-19), backtested on every Spain and
Argentina international since 2023, plus a **live layer hosted on Vercel**:
a serverless API polling the ESPN match feed that turns the score, clock,
red cards and shots into live win probabilities and fair player prop odds.

## Live app (Vercel)

```
index.html    dashboard - polls /api/live every 60s from the 3:00 PM EDT kickoff
api/live.py   serverless function (stdlib only, no API key needed):
              ESPN feed -> win prob + anytime-scorer props + total-goals lines
```

Deploy: push this repo to GitHub, then on vercel.com choose **Add New →
Project → Import** this repo and hit Deploy — zero config needed (static
`index.html` at root, Python function auto-detected under `api/`).

While the page is open it re-polls every 60 seconds ("collect every
minute"). The countdown shows before kickoff and polling picks the match up
automatically. Notes:

- Odds are **fair/no-vig model odds** for education — not a bookmaker feed.
- The model reacts to *key plays*: goals (via score), red cards (0.35x
  scoring rate for the short-handed team) and a bounded shots-on-target
  momentum multiplier.
- Server-side minute-by-minute *storage* (a persistent time series without
  the page open) needs Vercel Pro cron or an external pinger
  (e.g. cron-job.org hitting `/api/live`) plus Vercel KV - the endpoint is
  stateless by design so either can be bolted on.

## Pipeline

```
data/players.json   22 starters, club stats per season 2023-24 .. 2025-26
data/matches.csv    85 Spain/Argentina internationals 2023-2026 (90-min results + opponent Elo)
        |
src/scoring.py      player scores (0-100) -> team score = sum of 11 starters
src/backtest.py     fits team_score -> Elo-scale strength (ordered logistic), time-split validation
src/predict.py      Monte-Carlo final prediction (200k sims: 90 min + ET + pens)
src/live_model.py   gradient-boosted live win-probability model + demo
        |
output/             scores.json, model_params.json, final_prediction.json, live_model.joblib
```

Run in order:

```bash
cd src
python3 scoring.py && python3 backtest.py && python3 predict.py && python3 live_model.py
```

Requires `numpy pandas scipy scikit-learn joblib`.

## Player scoring

Each player-season scores 0-100 from:

- **Availability (45%)** — minutes vs a ~3400-min full workload
- **Performance (55%)** — (goals + 0.75×assists) per 90 vs a positional
  baseline (CB 0.07 … FW 0.65); goalkeepers use clean-sheet rate
- **League strength multiplier** — EPL 1.00, La Liga 0.98, … MLS 0.60
  (this is why Messi's elite raw numbers score ~56: MLS discount)

Seasons are recency-weighted (0.20 / 0.35 / 0.45). Team score = sum of the XI.

## Backtest

Team score maps to an Elo-scale strength (`alpha + beta*score`), opponents
use public Elo ratings, and an ordered-logistic draw band converts strength
difference to W/D/L probabilities. Trained on matches before 2025-07-01,
tested on the 24 matches after (including both teams' full WC 2026 runs):

| split | n | accuracy | Brier (3-class) | log loss |
|---|---|---|---|---|
| train 2023-01..2025-06 | 61 | 0.738 | 0.372 | 0.658 |
| test 2025-07..2026-07 | 24 | 0.833 | 0.308 | 0.558 |

Accuracy equals the "always predict a win" baseline (these teams almost
always win), so the model's value is probabilistic: Brier/log-loss beat
uniform (0.667 / 1.099) comfortably, and the fitted Elo-equivalents —
Spain ≈ 2197, Argentina ≈ 2146 — independently land within ~10 points of the
teams' real published Elo ratings, which the model was never shown.

## Live ML model (end goal)

`live_model.py` trains a `HistGradientBoostingClassifier` on 40k simulated
matches (minute-by-minute Poisson goals, red cards) using features a live
feed can supply: `minute, score_diff, elo_diff, red_card_diff,
lam_diff_remaining`. Holdout log-loss 0.598 vs naive 1.09-ish; the demo
prints a live win-probability trajectory for a sample final scenario.

**To go truly live:** replace `simulate_matches()` rows with historical
event-feed snapshots (API-Football, StatsBomb, Opta) and add live xG/shots
features — the model and interface stay identical.

## Known limitations (read before betting anything)

1. **Stats are approximate aggregates** compiled from public knowledge, and
   2025-26 club figures are full-season estimates. The pipeline is fully
   data-driven — swap `players.json` for exact FBref exports to upgrade.
2. Only Spain and Argentina carry player-sum scores, so the score→Elo slope
   `beta` is identified mostly by the gap between these two squads. Scoring
   more national squads is the highest-value next step.
3. Defensive quality is only proxied (GK clean sheets, CB/FB baselines);
   no xG, pressing, or defensive-action data.
4. Team score is held constant across the backtest window (current XI
   applied retroactively); real lineups rotated.
5. Spain's Round-of-32 result was not in sources retrieved and is omitted
   from `matches.csv`.
6. Lineups are the media-projected XIs as of the morning of the final.

## Prediction (2026-07-19)

- Team scores: **Spain 711.8 — Argentina 665.0**
- 90 minutes: **Spain 42.6% / Draw 27.6% / Argentina 29.7%**
- Trophy: **Spain 57.6% — Argentina 42.4%**
- Most likely scorelines: 1-1 (12.9%), Spain 1-0 (12.5%)
