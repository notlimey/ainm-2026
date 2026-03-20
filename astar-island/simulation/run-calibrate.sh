#!/bin/bash
# CMA-ES parameter calibration with multi-restart.
# Safe to run with nohup — survives terminal close.
#
# Usage:
#   ./run-calibrate.sh [name]          — run with defaults
#   nohup ./run-calibrate.sh pc2 &     — run in background, survives disconnect
#
# Two machines:
#   Machine 1: nohup ./run-calibrate.sh machine1 &
#   Machine 2: nohup ./run-calibrate.sh machine2 &
#   (different names = different random seeds = different search regions)
#
# Check progress:  tail -f data/calibration_machine1.log
# Stop:            kill %1   (or kill the PID)
# Params saved every 10 generations — safe to stop anytime.
#
# After calibration, test with:
#   ./simulate data/grids.bin 1 0 /dev/null --rollouts 200 --params data/params_machine1.bin

set -e
cd "$(dirname "$0")"

NAME=${1:-run1}
LOGFILE="data/calibration_${NAME}.log"
PARAMS="data/params_${NAME}.bin"

# Config
GENS_PER_RESTART=150
ROLLOUTS=30
NUM_RESTARTS=5

# Build
[ -f calibrate ] || c++ -std=c++17 -O3 -pthread -o calibrate calibrate.cpp
[ -f simulate ]  || c++ -std=c++17 -O3 -o simulate simulate.cpp

# Derive CMA-ES seed from name (so different names = different search)
CMA_BASE_SEED=$(echo -n "$NAME" | cksum | awk '{print $1 % 10000}')

{
echo "=========================================="
echo "  CMA-ES Calibration: $NAME"
echo "  $(date)"
echo "  Gens/restart: $GENS_PER_RESTART"
echo "  Rollouts: $ROLLOUTS"
echo "  Restarts: $NUM_RESTARTS"
echo "  Base seed: $CMA_BASE_SEED"
echo "  Output: $PARAMS"
echo "=========================================="
echo ""

# Restart 1: start from defaults with medium sigma
echo "=== RESTART 1/5 — from defaults, sigma=0.3, seed=$CMA_BASE_SEED ==="
./calibrate data/grids.bin data/ground_truth.bin \
    --generations $GENS_PER_RESTART --rollouts $ROLLOUTS \
    --output "$PARAMS" --sigma 0.3 --cma-seed $CMA_BASE_SEED
echo ""

# Restarts 2-5: warm-start from best so far, different sigmas and seeds
SIGMAS=(0.2 0.4 0.15 0.5)
for i in 1 2 3 4; do
    RESTART=$((i + 1))
    SIGMA=${SIGMAS[$((i-1))]}
    SEED=$((CMA_BASE_SEED + i * 1337))
    INIT_FLAG=""
    if [ -f "$PARAMS" ]; then
        INIT_FLAG="--init-params $PARAMS"
    fi
    echo "=== RESTART $RESTART/5 — warm-start, sigma=$SIGMA, seed=$SEED ==="
    ./calibrate data/grids.bin data/ground_truth.bin \
        --generations $GENS_PER_RESTART --rollouts $ROLLOUTS \
        --output "$PARAMS" --sigma "$SIGMA" --cma-seed "$SEED" $INIT_FLAG
    echo ""
done

echo "=========================================="
echo "  ALL RESTARTS COMPLETE"
echo "  $(date)"
echo "  Best params: $PARAMS"
echo "=========================================="

# Final validation with calibrated params
echo ""
echo "=== Final validation (300 rollouts) ==="
for R in 1 2 3 4 5 6 7; do
    for S in 0 1 2; do
        ./simulate data/grids.bin "$R" "$S" /dev/null \
            --rollouts 300 --params "$PARAMS" 2>&1 | grep "Score" | sed "s/^/  R${R}.S${S}: /"
    done
done

echo ""
echo "DONE! $(date)"

} 2>&1 | tee "$LOGFILE"
