#!/bin/bash
set -e

ROUND=${1:?Usage: ./predict-round.sh <round_number>}

echo "=== Fetching latest data ==="
cd aggregate-data
deno run --allow-net --allow-env --allow-read --allow-write fetch-analysis.ts

echo ""
echo "=== Converting JSON -> bin ==="
deno run --allow-read --allow-write convert-to-bin.ts

echo ""
echo "=== Building predictor ==="
cd ../simulation
g++ -std=c++17 -O2 -o predict predict.cpp

echo ""
echo "=== Predicting round $ROUND (all seeds) ==="
for SEED in 0 1 2 3 4; do
    OUT="data/pred_r${ROUND}_s${SEED}.bin"
    echo "  [r${ROUND}.s${SEED}]"
    ./predict data/training.bin data/grids.bin "$ROUND" "$SEED" "$OUT"
    echo ""
done

echo "=== Submitting ==="
cd ../aggregate-data
deno run --allow-net --allow-env --allow-read --allow-write submit-predictions.ts --round "$ROUND"

echo ""
echo "Done."
