#!/bin/bash
# auto-round.sh — Fully automated round handler for Astar Island
# Polls for active rounds, generates predictions from all models, oracle-selects best, submits.
# Usage: ./auto-round.sh [--loop]   (--loop polls every 60s for new rounds)
set -euo pipefail
cd "$(dirname "$0")"

SIM_DIR="simulation"
AGG_DIR="aggregate-data"
ROLLOUTS="${ROLLOUTS:-500}"
PARAMS="${PARAMS:-data/params.bin}"
CNN_CHANNELS="${CNN_CHANNELS:-16}"
CNN_EPOCHS="${CNN_EPOCHS:-30}"
QUERIES_PER_SEED="${QUERIES_PER_SEED:-10}"
VIEWPORT="${VIEWPORT:-15}"
TUNE_ROLLOUTS="${TUNE_ROLLOUTS:-30}"
TUNE_ITERS="${TUNE_ITERS:-50}"
MLP_EPOCHS="${MLP_EPOCHS:-15}"
BUCKET_EPOCHS="${BUCKET_EPOCHS:-7}"

log() { echo "[$(date '+%H:%M:%S')] $*"; }

# Get active round number from API
get_active_round() {
    cd "$AGG_DIR"
    deno run -A fetch-analysis.ts 2>/dev/null
    local round
    round=$(python3 -c "
import json
data = json.load(open('data/my-rounds.json'))
for r in data:
    if r.get('status') == 'active':
        print(r['round_number'])
        break
" 2>/dev/null || echo "")
    cd ..
    echo "$round"
}

# Process one round end-to-end
process_round() {
    local ROUND=$1
    log "=== Processing Round $ROUND ==="

    # 1. Convert data to binary
    log "Step 1: Converting data to binary..."
    cd "$AGG_DIR"
    deno run -A convert-to-bin.ts 2>/dev/null
    cd ..

    # 2. Generate sim predictions with base params (all 5 seeds)
    log "Step 2: Generating sim predictions ($ROLLOUTS rollouts)..."
    cd "$SIM_DIR"
    for S in 0 1 2 3 4; do
        ./simulate data/grids.bin "$ROUND" "$S" "data/pred_sim_r${ROUND}_s${S}.bin" \
            --rollouts "$ROLLOUTS" --params "$PARAMS" 2>/dev/null
        log "  sim seed $S: done"
    done

    # Copy sim as initial submission (safe fallback)
    for S in 0 1 2 3 4; do
        cp "data/pred_sim_r${ROUND}_s${S}.bin" "data/pred_r${ROUND}_s${S}.bin"
    done
    cd ..

    # 3. First submission (sim-only, ensures we have SOMETHING submitted)
    log "Step 3: Submitting sim-only predictions (safety net)..."
    cd "$AGG_DIR"
    deno run -A submit-predictions.ts --round "$ROUND" 2>&1
    cd ..

    # 4. Generate bucket + MLP predictions (fast, independent of queries)
    log "Step 4a: Generating bucket predictions..."
    cd "$SIM_DIR"
    for S in 0 1 2 3 4; do
        ./predict data/training.bin data/grids.bin "$ROUND" "$S" \
            "data/pred_bucket_r${ROUND}_s${S}.bin" --epochs "$BUCKET_EPOCHS" 2>/dev/null
    done
    log "  bucket: done (5 seeds)"

    log "Step 4b: Generating MLP predictions..."
    for S in 0 1 2 3 4; do
        ./mlp data/training.bin data/grids.bin "$ROUND" "$S" \
            "data/pred_mlp_r${ROUND}_s${S}.bin" --exclude "$ROUND" --epochs "$MLP_EPOCHS" 2>/dev/null
        log "  mlp seed $S: done"
    done
    cd ..

    # 5. Use queries
    log "Step 5: Using queries (${QUERIES_PER_SEED} per seed, viewport ${VIEWPORT})..."
    cd "$AGG_DIR"
    deno run -A query-round.ts "$ROUND" \
        --queries-per-seed "$QUERIES_PER_SEED" \
        --viewport "$VIEWPORT" \
        --model sim 2>&1
    cd ..

    # 6. Per-round parameter tuning using query observations
    log "Step 6: Tuning params for this round..."
    cd "$SIM_DIR"
    ./tune data/grids.bin "$ROUND" \
        --params "$PARAMS" \
        --queries-dir ../aggregate-data/data/queries \
        --output "data/params_r${ROUND}.bin" \
        --rollouts "$TUNE_ROLLOUTS" \
        --iterations "$TUNE_ITERS" 2>&1
    log "  tune: done"

    # 7. Generate predictions with tuned params
    log "Step 7: Generating predictions with tuned params..."
    for S in 0 1 2 3 4; do
        ./simulate data/grids.bin "$ROUND" "$S" "data/pred_tuned_r${ROUND}_s${S}.bin" \
            --rollouts "$ROLLOUTS" --params "data/params_r${ROUND}.bin" 2>/dev/null
        log "  tuned sim seed $S: done"
    done
    cd ..

    # 8. Generate blend predictions (uses all 4 model predictions + ground truth weights)
    log "Step 8: Generating blend predictions..."
    cd "$SIM_DIR"
    ./blend data/ground_truth.bin --data-dir data/ --grids data/grids.bin \
        --predict "$ROUND" 2>&1 | tail -8
    log "  blend: done"
    cd ..

    # 9. Oracle model selection — score all models against query observations, pick best per seed
    log "Step 9: Oracle model selection..."
    cd "$AGG_DIR"
    deno run -A select-best-model.ts "$ROUND" --data-dir ../simulation/data 2>&1
    cd ..

    # 10. Blend queries on top of best predictions
    log "Step 10: Query blending on best predictions..."
    cd "$AGG_DIR"
    deno run -A query-round.ts "$ROUND" --blend-only --model sim 2>&1
    cd ..

    # 11. Final submission
    log "Step 11: Final submission..."
    cd "$AGG_DIR"
    deno run -A submit-predictions.ts --round "$ROUND" 2>&1
    cd ..

    log "=== Round $ROUND complete ==="
}

# Main
if [[ "${1:-}" == "--loop" ]]; then
    log "Starting auto-round loop (polling every 60s)..."
    LAST_ROUND=0
    while true; do
        ROUND=$(get_active_round)
        if [ -n "$ROUND" ] && [ "$ROUND" != "$LAST_ROUND" ]; then
            process_round "$ROUND" || log "ERROR processing round $ROUND"
            LAST_ROUND=$ROUND
        fi
        sleep 60
    done
else
    ROUND="${1:-}"
    if [ -z "$ROUND" ]; then
        ROUND=$(get_active_round)
    fi
    if [ -z "$ROUND" ]; then
        log "No active round found"
        exit 1
    fi
    process_round "$ROUND"
fi
