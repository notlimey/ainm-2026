#!/bin/bash
set -e

ROUND=${1:?Usage: ./predict-round.sh <round_number> [--skip-queries] [--skip-submit]}
SKIP_QUERIES=false
SKIP_SUBMIT=false
MLP_EPOCHS=7

for arg in "$@"; do
    case $arg in
        --skip-queries) SKIP_QUERIES=true ;;
        --skip-submit) SKIP_SUBMIT=true ;;
    esac
done

echo "=========================================="
echo "  Round $ROUND — Full Pipeline"
echo "=========================================="

echo ""
echo "=== 1. Fetching latest data ==="
cd aggregate-data
deno run -A fetch-analysis.ts

echo ""
echo "=== 2. Converting JSON -> bin ==="
deno run -A convert-to-bin.ts

echo ""
echo "=== 3. Building predictors ==="
cd ../simulation
c++ -std=c++17 -O2 -o predict predict.cpp
c++ -std=c++17 -O2 -o mlp mlp.cpp

echo ""
echo "=== 4. Generating bucket predictions ==="
for SEED in 0 1 2 3 4; do
    echo "  [r${ROUND}.s${SEED}] bucket"
    ./predict data/training.bin data/grids.bin "$ROUND" "$SEED" "data/pred_r${ROUND}_s${SEED}.bin"
done

echo ""
echo "=== 5. Generating MLP predictions (${MLP_EPOCHS} epochs) ==="
for SEED in 0 1 2 3 4; do
    echo "  [r${ROUND}.s${SEED}] mlp"
    ./mlp data/training.bin data/grids.bin "$ROUND" "$SEED" "data/pred_mlp_r${ROUND}_s${SEED}.bin" \
        --exclude "$ROUND" --epochs "$MLP_EPOCHS"
done

echo ""
echo "=== 6. Query + Blend ==="
cd ../aggregate-data
if [ "$SKIP_QUERIES" = true ]; then
    echo "  Blending with stored queries only (--skip-queries)"
    deno run -A query-round.ts "$ROUND" --blend-only --model ensemble
else
    deno run -A query-round.ts "$ROUND" --model ensemble
fi

if [ "$SKIP_SUBMIT" = false ]; then
    echo ""
    echo "=== 7. Submitting ==="
    deno run -A submit-predictions.ts --round "$ROUND"
fi

echo ""
echo "=== 8. Generating preview ==="
deno run -A preview-prediction.ts

echo ""
echo "Done."
