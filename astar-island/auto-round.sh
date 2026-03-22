#!/bin/bash
# auto-round.sh — Fully automated round handler for Astar Island
# Polls for active rounds, generates predictions, tunes params, uses queries, submits.
# Usage: ./auto-round.sh [--loop]   (--loop polls every 60s for new rounds)
set -euo pipefail
cd "$(dirname "$0")"

SIM_DIR="simulation"
AGG_DIR="aggregate-data"
ROLLOUTS="${ROLLOUTS:-200}"
PARAMS="${PARAMS:-data/params.bin}"
CNN_CHANNELS="${CNN_CHANNELS:-16}"
CNN_EPOCHS="${CNN_EPOCHS:-30}"
QUERIES_PER_SEED="${QUERIES_PER_SEED:-10}"
VIEWPORT="${VIEWPORT:-10}"
TUNE_ROLLOUTS="${TUNE_ROLLOUTS:-30}"
TUNE_ITERS="${TUNE_ITERS:-50}"

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

    # 4. Use queries
    log "Step 4: Using queries (${QUERIES_PER_SEED} per seed, viewport ${VIEWPORT})..."
    cd "$AGG_DIR"
    deno run -A query-round.ts "$ROUND" \
        --queries-per-seed "$QUERIES_PER_SEED" \
        --viewport "$VIEWPORT" \
        --model sim 2>&1
    cd ..

    # 5. Per-round parameter tuning using query observations
    log "Step 5: Tuning params for this round..."
    cd "$SIM_DIR"
    ./tune data/grids.bin "$ROUND" \
        --params "$PARAMS" \
        --queries-dir ../aggregate-data/data/queries \
        --output "data/params_r${ROUND}.bin" \
        --rollouts "$TUNE_ROLLOUTS" \
        --iterations "$TUNE_ITERS" 2>&1
    log "  tune: done"

    # 6. Generate predictions with tuned params
    log "Step 6: Generating predictions with tuned params..."
    for S in 0 1 2 3 4; do
        ./simulate data/grids.bin "$ROUND" "$S" "data/pred_tuned_r${ROUND}_s${S}.bin" \
            --rollouts "$ROLLOUTS" --params "data/params_r${ROUND}.bin" 2>/dev/null
        log "  tuned sim seed $S: done"
    done
    cd ..

    # 7. Pick best (tuned vs base) per seed using query-observation score
    log "Step 7: Picking best predictions per seed..."
    cd "$AGG_DIR"
    deno eval "
import { readPrediction, entropy, NUM_CLASSES, terrainToClass } from './bin-io.ts';

const round = ${ROUND};
async function loadQ(s) {
    try { return JSON.parse(await Deno.readTextFile(\`data/queries/r\${round}/s\${s}_queries.json\`)); }
    catch { return []; }
}
function score(pred, queries, W, H) {
    const counts = Array.from({length:H},()=>Array.from({length:W},()=>new Array(${NUM_CLASSES:-6}).fill(0)));
    const samp = Array.from({length:H},()=>new Array(W).fill(0));
    for (const q of queries) {
        const {x,y,w,h} = q.viewport;
        for (let r=0;r<h&&y+r<H;r++) for (let c=0;c<w&&x+c<W;c++) {
            counts[y+r][x+c][terrainToClass(q.grid[r][c])]++;
            samp[y+r][x+c]++;
        }
    }
    let twkl=0, tent=0;
    for (let y=0;y<H;y++) for (let x=0;x<W;x++) {
        if (samp[y][x]<2) continue;
        const obs=counts[y][x].map(c=>c/samp[y][x]);
        const ent=entropy(obs);
        if (ent<1e-6) continue;
        let kl=0;
        for (let c=0;c<6;c++) if (obs[c]>1e-10) kl+=obs[c]*Math.log(obs[c]/Math.max(pred[y][x][c],1e-10));
        twkl+=ent*kl; tent+=ent;
    }
    return tent<1e-10?100:Math.max(0,Math.min(100,100*Math.exp(-3*twkl/tent)));
}

let baseTot=0, tunedTot=0, picked=[];
for (let s=0;s<5;s++) {
    const queries = await loadQ(s);
    const base = readPrediction(\`../simulation/data/pred_sim_r\${round}_s\${s}.bin\`);
    const tuned = readPrediction(\`../simulation/data/pred_tuned_r\${round}_s\${s}.bin\`);
    const bs = score(base.prediction, queries, base.W, base.H);
    const ts = score(tuned.prediction, queries, tuned.W, tuned.H);
    const winner = ts > bs ? 'tuned' : 'base';
    picked.push(winner);
    baseTot += bs; tunedTot += ts;
    console.log(\`  S\${s}: base=\${bs.toFixed(1)} tuned=\${ts.toFixed(1)} -> \${winner}\`);
    // Copy winner to submission path
    const src = winner === 'tuned'
        ? \`../simulation/data/pred_tuned_r\${round}_s\${s}.bin\`
        : \`../simulation/data/pred_sim_r\${round}_s\${s}.bin\`;
    await Deno.copyFile(src, \`../simulation/data/pred_r\${round}_s\${s}.bin\`);
}
console.log(\`  Avg: base=\${(baseTot/5).toFixed(1)} tuned=\${(tunedTot/5).toFixed(1)}\`);
console.log(\`  Picked: \${picked.join(', ')}\`);
" 2>&1
    cd ..

    # 8. Blend queries on top of best predictions
    log "Step 8: Query blending on best predictions..."
    cd "$AGG_DIR"
    deno run -A query-round.ts "$ROUND" --blend-only --model sim 2>&1
    cd ..

    # 9. Final submission
    log "Step 9: Final submission..."
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
