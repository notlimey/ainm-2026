#!/bin/bash
# Generate all 4 model predictions for historical rounds (1-8) × seeds (0-4)
# Used by blend.cpp to find optimal per-class ensemble weights
set -e
cd "$(dirname "$0")"

GRIDS="data/grids.bin"
GT="data/ground_truth.bin"
TRAIN="data/training.bin"
PARAMS="${1:-data/params.bin}"
EPOCHS="${2:-15}"

echo "=== Generating all model predictions ==="
echo "Params: $PARAMS"
echo "MLP epochs: $EPOCHS"
echo ""

# Discover which rounds have ground truth
ROUNDS=""
for R in $(seq 1 20); do
    # Check if grids.bin has this round
    ./simulate "$GRIDS" "$R" 0 /dev/null --rollouts 1 2>/dev/null && ROUNDS="$ROUNDS $R" || true
done
echo "Rounds with data: $ROUNDS"

for R in $ROUNDS; do
    for S in 0 1 2 3 4; do
        echo "--- Round $R, Seed $S ---"

        # Bucket predictor
        if [ ! -f "data/pred_bucket_r${R}_s${S}.bin" ]; then
            ./predict "$TRAIN" "$GRIDS" "$R" "$S" "data/pred_bucket_r${R}_s${S}.bin" --exclude "$R" --epochs 7 2>/dev/null
            echo "  bucket: done"
        else
            echo "  bucket: exists"
        fi

        # MLP (with new position features)
        if [ ! -f "data/pred_mlp_r${R}_s${S}.bin" ] || [ "$FORCE" = "1" ]; then
            ./mlp "$TRAIN" "$GRIDS" "$R" "$S" "data/pred_mlp_r${R}_s${S}.bin" --exclude "$R" --epochs "$EPOCHS" 2>/dev/null
            echo "  mlp: done"
        else
            echo "  mlp: exists"
        fi

        # Simulator
        if [ -f "$PARAMS" ]; then
            if [ ! -f "data/pred_sim_r${R}_s${S}.bin" ] || [ "$FORCE" = "1" ]; then
                ./simulate "$GRIDS" "$R" "$S" "data/pred_sim_r${R}_s${S}.bin" --rollouts 500 --params "$PARAMS" 2>/dev/null
                echo "  sim: done"
            else
                echo "  sim: exists"
            fi
        else
            echo "  sim: SKIPPED (no params file)"
        fi

        # CNN (uses sim predictions as input)
        if [ -f "data/pred_sim_r${R}_s${S}.bin" ]; then
            if [ ! -f "data/pred_cnn_r${R}_s${S}.bin" ] || [ "$FORCE" = "1" ]; then
                ./cnn "$GRIDS" "$GT" "$R" "$S" "data/pred_cnn_r${R}_s${S}.bin" --exclude "$R" --epochs 30 --sim-dir data/ 2>/dev/null
                echo "  cnn: done"
            else
                echo "  cnn: exists"
            fi
        else
            echo "  cnn: SKIPPED (no sim prediction)"
        fi
    done
done

echo ""
echo "=== Done! Run blend to find optimal weights ==="
echo "./blend data/ground_truth.bin --data-dir data/"
