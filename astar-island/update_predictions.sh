#!/bin/bash
set -e

# Pure local prediction + scoring. No API calls, no fetching, no submitting.
# Uses current data and current C++ code to generate all predictions and show scores.
#
# Usage: ./update_predictions.sh [--sim-params path/to/params.bin]

SIM_PARAMS=""
for arg in "$@"; do
    case "$arg" in
        --sim-params) shift; SIM_PARAMS="$1"; shift ;;
        --sim-params=*) SIM_PARAMS="${arg#*=}" ;;
    esac
done

echo "=========================================="
echo "  Update Predictions (local only)"
echo "=========================================="

cd simulation

# Build if needed
echo ""
echo "=== Building ==="
[ -f predict ] && [ predict -nt predict.cpp ] || { echo "  Compiling predict..."; c++ -std=c++17 -O3 -o predict predict.cpp; }
[ -f mlp ] && [ mlp -nt mlp.cpp ] || { echo "  Compiling mlp..."; c++ -std=c++17 -O3 -o mlp mlp.cpp; }
[ -f simulate ] && [ simulate -nt simulate.cpp ] || { echo "  Compiling simulate..."; c++ -std=c++17 -O3 -o simulate simulate.cpp; }
echo "  Done."

# Discover available rounds by probing grids.bin
ROUNDS=""
for R in $(seq 1 15); do
    if ./predict data/training.bin data/grids.bin "$R" 0 /dev/null --exclude "$R" 2>&1 | grep -q "Grid \["; then
        ROUNDS="$ROUNDS $R"
    fi
done
ROUNDS=$(echo $ROUNDS | xargs)
echo "Rounds: $ROUNDS"

echo ""
echo "=== Bucket predictions ==="
for R in $ROUNDS; do
    for S in 0 1 2 3 4; do
        ./predict data/training.bin data/grids.bin "$R" "$S" "data/pred_bucket_r${R}_s${S}.bin" --exclude "$R" 2>&1 | grep -E "Score" || true
    done
done

echo ""
echo "=== MLP predictions (7 epochs) ==="
for R in $ROUNDS; do
    for S in 0 1 2 3 4; do
        ./mlp data/training.bin data/grids.bin "$R" "$S" "data/pred_mlp_r${R}_s${S}.bin" --exclude "$R" --epochs 7 2>&1 | grep -E "Score" || true
    done
done

echo ""
echo "=== Simulator predictions (500 rollouts) ==="
SIM_FLAG=""
if [ -n "$SIM_PARAMS" ] && [ -f "$SIM_PARAMS" ]; then
    SIM_FLAG="--params $SIM_PARAMS"
    echo "  Using calibrated params: $SIM_PARAMS"
elif [ -f data/params.bin ]; then
    SIM_FLAG="--params data/params.bin"
    echo "  Using calibrated params: data/params.bin"
else
    echo "  Using default params (no calibration file found)"
fi
for R in $ROUNDS; do
    for S in 0 1 2 3 4; do
        ./simulate data/grids.bin "$R" "$S" "data/pred_sim_r${R}_s${S}.bin" --rollouts 500 $SIM_FLAG --ground-truth data/ground_truth.bin 2>&1 | grep -E "Score" || true
    done
done

# Generate scores and preview
echo ""
echo "=== Scores ==="
cd ../aggregate-data
deno run -A preview-prediction.ts 2>&1

echo ""
echo "Preview: file://$(pwd)/data/preview.html"
