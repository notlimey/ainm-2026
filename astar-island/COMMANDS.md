# Astar Island — Commands

All commands run from the project root.

## 1. Fetch data from API

```bash
cd aggregate-data

# Fetch my-rounds, initial states (active + completed), and analysis ground truth
deno run --allow-net --allow-env --allow-read --allow-write fetch-analysis.ts
```

## 2. Convert JSON to bin

```bash
cd aggregate-data

# Converts analysis + initial states -> training.bin, grids.bin, ground_truth.bin
deno run --allow-read --allow-write --allow-env convert-to-bin.ts
```

Output goes to `simulation/data/`.

## 3. Predict

```bash
cd simulation
g++ -std=c++17 -O2 -o predict predict.cpp

# Single seed (auto-validates if ground truth exists)
./predict data/training.bin data/grids.bin <round> <seed> [output.bin]

# All seeds for a round
for s in 0 1 2 3 4; do
  ./predict data/training.bin data/grids.bin <round> $s data/pred_r<round>_s${s}.bin
done
```

## 4. Submit predictions

```bash
cd aggregate-data

# Submit one seed
deno run --allow-net --allow-env --allow-read --allow-write submit-predictions.ts ../simulation/data/pred_r7_s0.bin

# Submit all seeds for a round
deno run --allow-net --allow-env --allow-read --allow-write submit-predictions.ts --round 7
```

## 5. Refine with queries (active rounds only)

```bash
cd aggregate-data

# Uses simulate queries to sample high-uncertainty cells and blend with model
# Requires prediction.bin to exist first (run step 3)
deno run --allow-net --allow-env --allow-read --allow-write query-and-refine.ts <round> <seed> [queries=10] [viewport=15]

# Example: refine round 7 seed 0 with 10 queries
deno run --allow-net --allow-env --allow-read --allow-write query-and-refine.ts 7 0 10
```

Overwrites the prediction.bin with the refined version. Submit after.

## 6. Full pipeline (one command)

```bash
./predict-round.sh <round_number>
```

Fetches data, converts, predicts all 5 seeds, submits.

## 6. Visualize analysis

```bash
cd aggregate-data
deno run --allow-read --allow-write --allow-env visualize-analysis.ts
open data/analysis-viewer.html
```

## 7. Test database (scraped step-by-step data)

```bash
cd tools
g++ -std=c++17 -O2 -o test_database test_database.cpp
./test_database ../simulation/data/database.bin
```
