# Astar Island — Commands

All commands run from the project root unless noted.

## Quick Round Workflow

```bash
# 1. Fetch & convert (new round data + previous round GT)
cd aggregate-data
deno run -A fetch-analysis.ts
deno run -A convert-to-bin.ts

# 2. Generate sim predictions (strongest model)
cd simulation
for S in 0 1 2 3 4; do
  ./simulate data/grids.bin <ROUND> $S data/pred_sim_r<ROUND>_s${S}.bin \
    --rollouts 2000 --params data/params.bin
done

# 3. Copy as submission base
for S in 0 1 2 3 4; do
  cp data/pred_sim_r<ROUND>_s${S}.bin data/pred_r<ROUND>_s${S}.bin
done

# 4. Use all 50 queries (free data!)
cd aggregate-data
deno run -A query-round.ts <ROUND> --queries-per-seed 10 --viewport 15 --model sim

# 5. Submit
deno run -A submit-predictions.ts --round <ROUND>

# 6. Dashboard
deno run -A preview-prediction.ts
open data/preview.html
```

## Compilation

```bash
cd simulation

# Mac
c++ -std=c++17 -O3 -o predict predict.cpp
c++ -std=c++17 -O3 -o mlp mlp.cpp
c++ -std=c++17 -O3 -o simulate simulate.cpp
c++ -std=c++17 -O3 -o calibrate calibrate.cpp
c++ -std=c++17 -O3 -o cnn cnn.cpp
c++ -std=c++17 -O3 -o blend blend.cpp

# Linux (Ryzen) — must add -pthread
g++ -std=c++17 -O3 -pthread -o simulate simulate.cpp
g++ -std=c++17 -O3 -pthread -o calibrate calibrate.cpp
g++ -std=c++17 -O3 -pthread -o cnn cnn.cpp
```

## Individual Model Commands

### Simulator (best model, ~74 avg)
```bash
./simulate data/grids.bin <round> <seed> output.bin --rollouts 2000 --params data/params.bin
```

### Bucket predictor (~67 avg)
```bash
./predict data/training.bin data/grids.bin <round> <seed> output.bin --epochs 7
```
**Note:** Arg order is `training.bin` first, then `grids.bin`.

### MLP (~66 avg)
```bash
./mlp data/grids.bin data/ground_truth.bin <round> <seed> output.bin --exclude <round> --epochs 15
```

### CNN / U-Net (~71 avg)
```bash
# Requires sim predictions to exist first!
./cnn data/grids.bin data/ground_truth.bin <round> <seed> output.bin \
  --exclude <round> --epochs 30 --sim-dir data [--threads N] [--channels 16|32]
```
- `--channels 16` = slim/fast (Mac default)
- `--channels 32` = full U-Net (Ryzen)
- `--threads` auto-detects cores

### Blend optimizer (cross-validated weights)
```bash
./blend data/ground_truth.bin --data-dir data/
```

### Calibrate sim params (run on Ryzen)
```bash
./calibrate data/grids.bin data/ground_truth.bin \
  --generations 300 --rollouts 50 --threads 32 \
  --seeds-per-round 3 --output data/params.bin
```

## Queries (50 per round, free!)
```bash
cd aggregate-data
deno run -A query-round.ts <round> --queries-per-seed 10 --viewport 15 --model sim
```
Targets high-entropy cells, Bayesian blends with model. Writes to `pred_r<N>_s<S>.bin`.

## Ryzen (fish shell)

```fish
# Generate all sim predictions
for R in 1 2 3 4 5 6 7 8 9 10
  for S in 0 1 2 3 4
    ./simulate data/grids.bin $R $S data/pred_sim_r{$R}_s{$S}.bin --rollouts 2000 --params data/params.bin
  end
end

# Generate all CNN predictions (needs sim predictions)
for R in 1 2 3 4 5 6 7 8 9 10
  for S in 0 1 2 3 4
    ./cnn data/grids.bin data/ground_truth.bin $R $S data/pred_cnn_r{$R}_s{$S}.bin \
      --exclude $R --epochs 30 --sim-dir data --threads 32 --channels 32
  end
end
```
