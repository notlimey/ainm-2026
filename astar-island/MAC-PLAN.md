# Mac Action Plan

Work to do on Mac M4 while Ryzen handles recalibration + batch prediction generation.

## Priority Order

### 1. Improve tune.cpp reliability (~30 min)

Current tune.cpp helps on 4/7 rounds but hurts on 3/7. Fixes:

- **Add regularization** — penalize deviation from base params so tuning stays conservative
- **More params** — add trade_range, port_prob, food_per_coastal, ruin_reclaim_range, collapse_defense
- **Higher rollouts** — 30→50 during tuning for less noise
- **Better stopping** — if no improvement after 15 iterations, stop early

Test: rerun backtest on R9/R10/R14/R17/R18/R19/R20, verify no round gets worse AND hard rounds still improve.

### 2. Use settlement properties from queries (~45 min)

The query API returns `settlements[].{population, food, wealth, defense, tech, owner_id}` but we throw this away! These directly reveal hidden param effects:

- High population → low raid_damage, high food_per_forest
- Low food across settlements → high winter_base_loss
- Many owner_ids → low conquest_prob
- High tech → high tech_diffusion / more trade

**Approach:** After querying, compute aggregate settlement statistics. Use these to constrain tune.cpp's search space or directly adjust params before tuning. E.g., if avg population is very high, shift expansion_pop and growth_rate toward higher values.

### 3. Improve query strategy (~20 min)

Current: 10 queries/seed, viewport 10×10.

Better allocation:
- First submit with base params (safety net)
- Use 30 queries (6/seed) with viewport 15×15 for maximum coverage → ~60% of cells observed
- Tune params
- Use remaining 20 queries (4/seed) on highest-uncertainty cells with smaller viewport 8×8

### 4. Auto-round.sh improvements (~15 min)

- Add settlement property extraction step
- Split query budget: 30 for tuning + 20 for blending
- Add logging to file for debugging overnight failures
- Add retry logic for API failures

### 5. Test higher rollouts (~10 min)

Quick test: does 500 rollouts vs 200 make a measurable difference?

```bash
cd simulation
# R15 seed 0: baseline at 200 rollouts
./simulate data/grids.bin 15 0 /dev/null --rollouts 200 --params data/params.bin
# R15 seed 0: 500 rollouts
./simulate data/grids.bin 15 0 /dev/null --rollouts 500 --params data/params.bin
# R15 seed 0: 1000 rollouts
./simulate data/grids.bin 15 0 /dev/null --rollouts 1000 --params data/params.bin
```

## What NOT to do on Mac

- **Don't recalibrate globally** — Mac CMA-ES with 30 rollouts/8 threads produces worse params than Ryzen
- **Don't train CNN 32ch** — Mac 16ch is already tested (and 32ch was worse on Mac, likely overfitting)
- **Don't regenerate all predictions** — Wait for Ryzen to do this with better params

## When Ryzen finishes

1. Pull new params.bin from Ryzen
2. Regenerate R22 (active round) predictions with new params
3. Run tune.cpp on R22 with new base params
4. Submit best predictions
