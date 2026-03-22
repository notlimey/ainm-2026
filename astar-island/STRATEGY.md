# Astar Island — Strategy & Status

## Current Position
- **Rank:** #148 (ws=229.3)
- **Best weighted:** R21 = 82.3 × 2.79 = 229.3
- **Hot streak:** 67.3 (last 3 rounds avg)
- **Top teams:** ws ~260, hot streak ~92-94

## Score History

| Round | Score | Weight | WS | Notes |
|-------|-------|--------|-----|-------|
| R7 | 59.8 | 1.41 | 84 | |
| R8 | 71.6 | 1.48 | 106 | |
| R9 | **90.9** | 1.55 | 141 | Best raw score |
| R10 | 68.7 | 1.63 | 112 | |
| R14 | 57.3 | 1.98 | 114 | |
| R15 | **85.5** | 2.08 | 178 | |
| R16 | **80.6** | 2.18 | 176 | |
| R17 | 72.8 | 2.29 | 167 | |
| R18 | **47.7** | 2.41 | 115 | Worst round |
| R19 | 58.9 | 2.53 | 149 | |
| R20 | 60.7 | 2.65 | 161 | Seed 4 missed (0.0) |
| R21 | **82.3** | 2.79 | 229 | Tuned params helped |

**Average: 69.7** | Good (80+): 4 rounds | Bad (<65): 5 rounds

## What Matters for Rank

Leaderboard = `max(round_score × round_weight)` across all rounds. Later rounds have higher weights (1.05^N).

| Target Score | On R25 (w≈3.4) | On R30 (w≈4.3) | Rank |
|---|---|---|---|
| 80 | ws=272 | ws=344 | ~1 |
| 85 | ws=289 | ws=366 | ~1 |
| 90 | ws=306 | ws=387 | ~1 |

**Conclusion:** Even a score of 80 on a late round (R25+) would top the leaderboard. We need consistency, not perfection.

## Root Cause Analysis

### Why we sometimes score 90+ (R9, R15, R21)
- Hidden params happen to match our global calibration
- Low settlement count (28-32 initial) = less chaos
- Trade/expansion dynamics within our model's sweet spot

### Why we sometimes score <60 (R14, R18, R19)
- Hidden params diverge significantly from our global calibration
- High settlement count (51-61) = more expansion/collapse dynamics
- Settlement survival rate differs from what our sim predicts

### The core problem
We use ONE global param set for ALL rounds. The organizers use DIFFERENT hidden params per round. When our params are close → 85+. When they're far → 50.

**Top teams likely:** have a more accurate simulator, do per-round adaptation, or use ML to learn corrections.

## Architecture & Models

| Model | CV Avg | Role |
|---|---|---|
| **Simulator** | 69.1 | Core model — Monte Carlo Norse world sim, 40 params |
| **CNN/U-Net** | 66.3 | Learns corrections on sim output |
| **MLP** | 65.7 | 23 hand-crafted features per cell |
| **Bucket** | 65.8 | Histogram from training data |
| **Blend** | 69.5 | Per-class weighted ensemble (+0.5 over sim) |

**Blend barely helps.** The sim IS the model. Improving sim = improving score.

## Key Tools

- `simulate.cpp` — Monte Carlo sim with param jitter
- `calibrate.cpp` — CMA-ES global param optimizer (needs Ryzen)
- `tune.cpp` — **NEW** Per-round param tuning using query observations (Nelder-Mead)
- `auto-round.sh` — Full automation: fetch → sim → submit → query → tune → pick-best → resubmit

## Recent Changes (March 22)

1. **Trade fix** — Changed from "same faction only" to "not at war". Factions that raided each other this year can't trade, all others can.
2. **tune.cpp** — Per-round parameter tuning. Uses 50 query observations to optimize 8 key sim params for the specific round via Nelder-Mead.
3. **auto-round.sh** — Overnight automation with tune + pick-best strategy.

### tune.cpp backtest results

| Round | Base | Tuned | Diff | Winner |
|---|---|---|---|---|
| R9 | 84.7 | 80.7 | -4.0 | base |
| R10 | 56.8 | **75.8** | **+19.0** | tuned |
| R14 | 56.5 | **62.8** | +6.3 | tuned |
| R17 | 70.6 | **81.0** | **+10.4** | tuned |
| R18 | 35.9 | **52.7** | **+16.8** | tuned |
| R19 | 55.1 | 47.6 | -7.4 | base |
| R20 | 67.9 | 43.2 | -24.7 | base |

**Strategy:** Always generate both, pick best per seed based on query-observation score.

## Priority Improvements

### Tier 1 — Do Now (highest impact)

1. **Ryzen recalibration** — Recalibrate global params with trade fix + 21 rounds GT on Ryzen. Current params were calibrated on old trade code with fewer rounds. (See RYZEN.md)

2. **Improve tune.cpp** — Make per-round tuning more reliable:
   - More rollouts during tuning (30→50)
   - Add regularization to penalize large deviations from base
   - Tune more params (add trade_range, port_prob, ruin_reclaim_range)
   - Use viewport 15 (not 10) for more cell coverage

3. **Use settlement properties from queries** — The API returns population, food, wealth, defense, tech, owner_id for each settlement in the viewport. We're NOT using this. These directly reveal hidden param effects.

### Tier 2 — Medium Impact

4. **500-1000 rollouts** for final predictions (currently 200). GT uses "hundreds".

5. **Better query budget allocation:**
   - 30 queries for tuning (6/seed, spread across map)
   - 20 queries for Bayesian blending on highest-uncertainty cells

6. **Wealth mechanics** — In our sim, wealth accumulates but doesn't affect anything. May affect defense/growth in real sim.

7. **CNN on Ryzen** — 32ch, 50 epochs. Our Mac 16ch CNN is weak.

### Tier 3 — Experimental

8. **Multi-start tuning** — 3 random starting perturbations, pick best.
9. **Ensemble tuned+base** — Blend instead of pick-one.
10. **Settlement state inference** — Use settlement properties to directly estimate hidden params.
