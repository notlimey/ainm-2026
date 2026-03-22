# Astar Island — Claude Project Guide

## What This Is
NM i AI (Norwegian AI Championship) challenge: predict probability distributions over 6 terrain classes for every cell in a 40×40 grid after 50 simulated years of Norse world simulation. Scored via entropy-weighted KL divergence.

## Current Status (R21 submitted with tuned params)

### Score Trajectory
- R7: 59.8 — bucket+MLP only
- R8: 71.6 — bucket+MLP+sim uncalibrated
- R9: 90.9 — calibrated sim + queries (best raw score)
- R10: 68.7 — sim + blend + queries
- R11-R13: missed
- R14: 57.3 — sim + blend + queries
- R15: 85.5 — best weighted score (177.8)
- R16: 80.6
- R17: 72.8
- R18: 47.7 — worst round (hard round)
- R19: 58.9
- R20: 60.7 — seed 4 missed (0.0), would have been ~75
- R21: submitted with per-round tuned params

### Models (avg cross-validation score, 13 rounds, 50 rollouts)
1. **CNN/U-Net** (69.8 avg) — Learns corrections on top of sim. Excels on R4 (90+), mountains. Only R1-R10 predictions available.
2. **Simulator** (varies) — Monte Carlo Norse world sim, CMA-ES calibrated, 50 params (40 original + 10 v2). Best on most rounds but terrible on R12 (35).
3. **Bucket** — Histogram predictor from training data. Best for empty/plains class.
4. **MLP** — 2-layer neural net with 23 hand-crafted features per cell.

### Per-Class Best Model (from blend optimizer, 13 rounds)
- Empty/Plains: Bucket 100%
- Settlement: MLP 10% + Sim 90%
- Port: Sim 100%
- Ruin: CNN 100%
- Forest: Sim 60% + CNN 40%
- Mountain: CNN 100%

## Architecture / Key Files

### C++ (in `simulation/`)
- `simulate.cpp` — Monte Carlo simulator. Runs N rollouts, aggregates terrain counts → probabilities.
  - `./simulate data/grids.bin <round> <seed> output.bin --rollouts 2000 --params data/params.bin`
  - Compile: `c++ -std=c++17 -O3 -o simulate simulate.cpp` (add `-pthread` on Linux)
- `calibrate.cpp` — CMA-ES parameter optimizer for simulator. Trains on ground truth data.
  - `./calibrate data/grids.bin data/ground_truth.bin --generations 300 --rollouts 50 --threads 32 --output data/params.bin`
  - Compile: `g++ -std=c++17 -O3 -pthread -o calibrate calibrate.cpp`
  - Outputs `params.bin` (~164 bytes, SIMP format)
- `cnn.cpp` — U-Net (2-level, skip connections, instance norm, residual learning, 8× data augmentation)
  - `./cnn data/grids.bin data/ground_truth.bin <round> <seed> output.bin --exclude <round> --epochs 30 --sim-dir data --threads 32 --channels 32`
  - `--channels 16` = slim/fast (Mac), `--channels 32` = full (Ryzen)
  - Requires sim predictions as input (reads `pred_sim_r{R}_s{S}.bin`)
  - Uses im2col + BLAS GEMM for convolutions (11× faster than naive loops)
  - Compile: `c++ -std=c++17 -O3 -framework Accelerate -o cnn cnn.cpp` (Mac)
  - Compile: `g++ -std=c++17 -O3 -pthread -lopenblas -o cnn cnn.cpp` (Linux)
- `predict.cpp` — Bucket predictor.
  - `./predict data/training.bin data/grids.bin <round> <seed> output.bin --epochs 7`
  - Note: arg order is `training.bin` first, then `grids.bin` (matches usage string in code).
- `mlp.cpp` — MLP predictor with features from `features.hpp`.
  - `./mlp data/training.bin data/grids.bin <round> <seed> output.bin --exclude <round> --epochs 15`
  - Note: arg order is `training.bin` first, then `grids.bin` (same as predict.cpp).
- `blend.cpp` — Per-class ensemble weight optimizer with LOOCV.
  - `./blend data/ground_truth.bin --data-dir data/ --grids data/grids.bin`
  - `--grids` enables terrain-aware smart flooring (+1.1 CV points)
- `tune.cpp` — Per-round parameter tuning using query observations.
  - `./tune data/grids.bin <round> --params data/params.bin --queries-dir ../aggregate-data/data/queries --output data/params_r{R}.bin`
  - Uses Nelder-Mead to optimize 12 params against query observations (terrain KL)
  - **+19 points on hard rounds (R10, R17, R18), slight risk on easy rounds → use with pick-best strategy**
  - Always generate both base and tuned predictions, score against query observations, pick the better one
- `features.hpp` — Feature extraction (23 features per cell incl. x/W, y/H position)
- `sim_params.hpp` — Shared SimParams struct (used by simulate.cpp + calibrate.cpp)
- `io.hpp` — Shared I/O utilities (load_grid, load_ground_truth, write_prediction, apply_floor, score_prediction)

### TypeScript (in `aggregate-data/`)
- `fetch-analysis.ts` — Fetches ground truth + initial grids from API
- `convert-to-bin.ts` — Converts JSON → binary (training.bin, grids.bin, ground_truth.bin)
- `query-round.ts` — Sends viewport queries, Bayesian blends with model predictions
- `select-best-model.ts` — Oracle model selection: scores all models against query observations, picks best per seed
- `submit-predictions.ts` — Submits prediction .bin files to API
- `preview-prediction.ts` — Generates HTML dashboard with all models + scores
- `client.ts` — API client wrapper

### Binary Formats
- **ASTP** (predictions): magic(4) + ver(2) + round(4) + seed(4) + W(4) + H(4) + float[H][W][6]
- **SIMP** (sim params): magic(4) + ver(2) + float[N_params]
- **grids.bin**: magic(4) + ver(2) + count(4) + entries[round(4) + seed(4) + W(4) + H(4) + int[H][W]]
- **ground_truth.bin**: magic(4) + ver(2) + count(4) + entries[round(4) + seed(4) + W(4) + H(4) + float[H][W][6]]

## Standard Workflow Per Round

**Automated (recommended):**
```bash
./auto-round.sh          # Process current active round
./auto-round.sh --loop   # Poll every 60s for new rounds (overnight mode)
```

**Manual workflow:**
```bash
# 1. Fetch + convert
cd aggregate-data && deno run -A fetch-analysis.ts && deno run -A convert-to-bin.ts

# 2. Sim predictions (base params)
cd simulation
for S in 0 1 2 3 4; do
  ./simulate data/grids.bin <R> $S data/pred_sim_r<R>_s${S}.bin --rollouts 200 --params data/params.bin
done
for S in 0 1 2 3 4; do cp data/pred_sim_r<R>_s${S}.bin data/pred_r<R>_s${S}.bin; done

# 3. First submit (safety net)
cd aggregate-data && deno run -A submit-predictions.ts --round <R>

# 4. Queries
deno run -A query-round.ts <R> --queries-per-seed 10 --viewport 15 --model sim

# 5. Per-round tune (uses query observations to optimize params for THIS round)
cd simulation
./tune data/grids.bin <R> --params data/params.bin --queries-dir ../aggregate-data/data/queries \
    --output data/params_r<R>.bin --rollouts 30 --iterations 50

# 6. Generate tuned predictions
for S in 0 1 2 3 4; do
  ./simulate data/grids.bin <R> $S data/pred_tuned_r<R>_s${S}.bin --rollouts 200 --params data/params_r<R>.bin
done

# 7. Pick best (tuned vs base) per seed, blend queries, resubmit
cd aggregate-data && deno run -A submit-predictions.ts --round <R>
```

## Ryzen Machine (Linux, fish shell)

```fish
# Compile
g++ -std=c++17 -O3 -pthread -o simulate simulate.cpp
g++ -std=c++17 -O3 -pthread -o calibrate calibrate.cpp
g++ -std=c++17 -O3 -pthread -o cnn cnn.cpp

# Sim predictions
for R in 1 2 3 4 5 6 7 8 9 10
  for S in 0 1 2 3 4
    ./simulate data/grids.bin $R $S data/pred_sim_r{$R}_s{$S}.bin --rollouts 2000 --params data/params.bin
  end
end

# CNN predictions (needs sim predictions as input)
for R in 1 2 3 4 5 6 7 8 9 10
  for S in 0 1 2 3 4
    ./cnn data/grids.bin data/ground_truth.bin $R $S data/pred_cnn_r{$R}_s{$S}.bin \
      --exclude $R --epochs 30 --sim-dir data --threads 32 --channels 32
  end
end

# Calibrate (re-run when new GT data available)
./calibrate data/grids.bin data/ground_truth.bin \
  --generations 300 --rollouts 50 --threads 32 \
  --seeds-per-round 3 --output data/params_v2.bin
```

## Recent Improvements (March 22)

### Viewport 15×15 (was 10×10)
- API allows max 15×15 per query. Previous code used 10×10 = 100 cells. Now 15×15 = 225 cells.
- **2.25× more data per query** from same 50-query budget.
- Changed in: `auto-round.sh` (default), `query-round.ts` (default + usage string).

### Settlement Property Inference
Query API returns settlement properties (population, food, wealth, defense, owner_id, has_port) — previously stored as `unknown[]` and ignored.

**query-round.ts changes:**
- Typed `QuerySettlement` interface with all settlement fields
- `aggregateSettlementStats()` aggregates stats across all queries for a round
- Settlement stats printed during pooled temperature scaling pass
- Temperature optimization now also searches Port/Ruin temps (5 classes, not just 3)
- 2-stage temp optimization: coarse grid + fine refinement (±0.05 steps)

**tune.cpp changes:**
- Tunes 12 params with Nelder-Mead using query observations (terrain KL only)
- Params tuned: expansion_pop, expansion_prob, growth_rate, raid_prob_base, winter_base_loss, collapse_pop, ruin_reclaim_prob, food_per_forest + v2: expansion_split, growth_food_cost, food_cap, defense_recovery

## Known Issues / TODO
1. ~~**predict.cpp grid loading bug**~~ — FIXED: was a documentation bug. Arg order is `training.bin grids.bin`, not the reverse.
2. ~~**Sim too entropic on empty cells**~~ — FIXED: smart probability floor. Only floors reachable classes (no fake mountain/ruin on static cells). **+2.0 avg improvement across all rounds.**
3. **R3 is terrible everywhere** (39-51) — Only 420 dynamic cells/seed. Settlements completely collapse in GT (1.8% survival) but sim over-predicts them. Fundamentally different hidden params than average.
4. ~~**expansion_range added to calibrate.cpp**~~ — Now 50 params total (40 original + 10 new v2 params). Needs recalibration.
5. **CNN instance norm**: Must use training mode for inference when trained with threads (running stats not synced).
6. **Settlement under-prediction is 88.7% of total error** — Addressed by new v2 params: expansion_split, growth_food_cost, food_cap, wealth_growth_bonus. Needs recalibration to take effect.
7. ~~**~15 hardcoded magic numbers in simulate.cpp**~~ — 8 exposed as calibration params + 2 wealth interaction params. ~25 more remain (see review notes).
8. ~~**Wealth mechanic**~~ — Now active: wealth_growth_bonus (wealth boosts growth), wealth_defense_bonus (wealth boosts defense recovery). Default 0.0 until calibrated.
9. **Ryzen recalibration needed** — CRITICAL: must recalibrate with 50 params to unlock v2 improvements. Command: `./calibrate data/grids.bin data/ground_truth.bin --generations 400 --rollouts 50 --threads 32 --output data/params_v3.bin`
10. **Oracle model selection** — `select-best-model.ts` scores all models against queries, picks best per seed. Integrated into auto-round.sh. On historical data, blend wins almost every seed.

## Scoring Formula
```
score = max(0, min(100, 100 * exp(-3 * weighted_kl)))
weighted_kl = sum(entropy_i * kl_i) / sum(entropy_i)  // only dynamic cells (entropy > 1e-6)
```

## User Preferences
- Prefers C++ over Python for all compute
- Has Mac M4 (dev) + Ryzen 9 5950X / RTX 3090 Ti (Linux, compute)
- Linux uses fish shell, not bash
- Don't run tasks > 60 seconds — give commands for user to run and monitor
- GPU not useful for this workload (branching grid logic, small 40×40 grids)
