#!/bin/bash
# Full pipeline for a round: fetch → convert → sim → queries → submit → dashboard.
# Run from project root.
#
# Usage:
#   ./prepare_for_round.sh <ROUND> [options]
#
# Options:
#   --rollouts N       Sim rollouts (default: 2000)
#   --skip-fetch       Skip fetching from API (use existing data)
#   --skip-queries     Skip API queries, blend with stored queries only
#   --skip-submit      Don't submit (dry run)
#   --smart-alloc      Distribute queries proportionally by seed entropy
#   --queries N        Queries per seed (default: 10, total = N × 5 seeds)
#   --viewport N       Query viewport size (default: 15)
set -e
cd "$(dirname "$0")"

ROUND=${1:?Usage: ./prepare_for_round.sh <ROUND> [--rollouts N] [--skip-fetch] [--skip-queries] [--skip-submit] [--smart-alloc]}
shift

# Defaults
ROLLOUTS=1000
SKIP_FETCH=false
SKIP_QUERIES=false
SKIP_SUBMIT=false
SMART_ALLOC=""
QUERIES=10
VIEWPORT=10

# Parse options
while [[ $# -gt 0 ]]; do
    case $1 in
        --rollouts) ROLLOUTS="$2"; shift 2 ;;
        --skip-fetch) SKIP_FETCH=true; shift ;;
        --skip-queries) SKIP_QUERIES=true; shift ;;
        --skip-submit) SKIP_SUBMIT=true; shift ;;
        --smart-alloc) SMART_ALLOC="--smart-alloc"; shift ;;
        --queries) QUERIES="$2"; shift 2 ;;
        --viewport) VIEWPORT="$2"; shift 2 ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

echo "=========================================="
echo "  Round $ROUND — Prepare & Submit"
echo "=========================================="
echo "  Rollouts: $ROLLOUTS"
echo "  Queries: ${QUERIES}/seed ($(($QUERIES * 5)) total)"
echo "  Viewport: ${VIEWPORT}×${VIEWPORT}"
[ "$SKIP_FETCH" = true ] && echo "  Skipping fetch"
[ "$SKIP_QUERIES" = true ] && echo "  Skipping queries"
[ "$SKIP_SUBMIT" = true ] && echo "  Skipping submit"
[ -n "$SMART_ALLOC" ] && echo "  Smart query allocation: ON"
echo ""

# ── Step 1: Fetch & convert ──
if [ "$SKIP_FETCH" = false ]; then
    echo "=== 1. Fetching latest data ==="
    cd aggregate-data
    deno run -A fetch-analysis.ts
    echo ""
    echo "=== 2. Converting to binary ==="
    deno run -A convert-to-bin.ts
    cd ..
else
    echo "=== 1-2. Skipped fetch/convert ==="
fi

# ── Step 2: Check binaries ──
echo ""
echo "=== 3. Checking binaries ==="
cd simulation
if [ ! -f simulate ]; then
    echo "  ERROR: simulate not found. Run ./setup.sh first."
    exit 1
fi
echo "  simulate: OK"

# ── Step 3: Sim predictions ──
echo ""
echo "=== 4. Sim predictions ($ROLLOUTS rollouts) ==="
SIM_FLAGS=""
if [ -f data/params.bin ]; then
    SIM_FLAGS="--params data/params.bin"
    echo "  Using calibrated params"
else
    echo "  WARNING: No params.bin, using defaults"
fi

for SEED in 0 1 2 3 4; do
    OUT="data/pred_sim_r${ROUND}_s${SEED}.bin"
    echo -n "  [R${ROUND}.S${SEED}] "
    ./simulate data/grids.bin "$ROUND" "$SEED" "$OUT" --rollouts "$ROLLOUTS" $SIM_FLAGS
done

# ── Step 4: Bucket + MLP + CNN ──
echo ""
echo "=== 4b. Bucket predictions ==="
for SEED in 0 1 2 3 4; do
    echo -n "  [R${ROUND}.S${SEED}] "
    ./predict data/training.bin data/grids.bin "$ROUND" "$SEED" \
        "data/pred_bucket_r${ROUND}_s${SEED}.bin" --epochs 7
done

echo ""
echo "=== 4c. MLP predictions ==="
for SEED in 0 1 2 3 4; do
    echo -n "  [R${ROUND}.S${SEED}] "
    ./mlp data/training.bin data/grids.bin "$ROUND" "$SEED" \
        "data/pred_mlp_r${ROUND}_s${SEED}.bin" --exclude "$ROUND" --epochs 15
done

echo ""
echo "=== 4d. CNN predictions (slim) ==="
for SEED in 0 1 2 3 4; do
    echo -n "  [R${ROUND}.S${SEED}] "
    ./cnn data/grids.bin data/ground_truth.bin "$ROUND" "$SEED" \
        "data/pred_cnn_r${ROUND}_s${SEED}.bin" --exclude "$ROUND" \
        --epochs 30 --sim-dir data --channels 16
done

# ── Step 5: Blend ──
echo ""
echo "=== 5. Ensemble blend ==="
./blend data/ground_truth.bin --data-dir data/ --grids data/grids.bin --predict "$ROUND"

cd ..

# ── Step 5: Queries ──
echo ""
echo "=== 6. Queries + Bayesian blending ==="
cd aggregate-data
if [ "$SKIP_QUERIES" = true ]; then
    echo "  Blending with stored queries only"
    deno run -A query-round.ts "$ROUND" --blend-only --model blend --queries-per-seed "$QUERIES" --viewport "$VIEWPORT" $SMART_ALLOC
else
    deno run -A query-round.ts "$ROUND" --model blend --queries-per-seed "$QUERIES" --viewport "$VIEWPORT" $SMART_ALLOC
fi

# ── Step 6: Submit ──
if [ "$SKIP_SUBMIT" = false ]; then
    echo ""
    echo "=== 7. Submitting ==="
    deno run -A submit-predictions.ts --round "$ROUND"
else
    echo ""
    echo "=== 7. Skipped submit ==="
fi

# ── Step 7: Dashboard ──
echo ""
echo "=== 8. Updating dashboard ==="
deno run -A preview-prediction.ts

echo ""
echo "=========================================="
echo "  Round $ROUND complete!"
echo "=========================================="
echo "  Dashboard: open aggregate-data/data/preview.html"
