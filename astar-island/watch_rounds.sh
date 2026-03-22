#!/bin/bash
# Autonomous round watcher — runs overnight, auto-submits predictions.
#
# What it does every cycle:
#   1. Checks API for active round
#   2. If new round found → fetch, sim, query, submit (full pipeline)
#   3. If already submitted but queries remain → use remaining queries + re-submit
#   4. If round completed → fetch ground truth for scoring data
#   5. Sleep until next check
#
# Usage:
#   ./watch_rounds.sh                     # check every 150 min (2.5 hours)
#   ./watch_rounds.sh --interval 60       # check every 60 min
#   ./watch_rounds.sh --rollouts 3000     # more sim rollouts
#   nohup ./watch_rounds.sh &             # run in background
#
# Logs: watch_rounds.log (timestamped, appended)
# State: .watch_state (tracks processed rounds)
set -eo pipefail
cd "$(dirname "$0")"

# ── Config ──
INTERVAL_MIN=150       # check interval in minutes
ROLLOUTS=1000           # sim rollouts (1000 = good balance on Mac M4)
VIEWPORT=10
QUERIES_PER_SEED=10
LOG="watch_rounds.log"
STATE=".watch_state"

while [[ $# -gt 0 ]]; do
    case $1 in
        --interval) INTERVAL_MIN="$2"; shift 2 ;;
        --rollouts) ROLLOUTS="$2"; shift 2 ;;
        --viewport) VIEWPORT="$2"; shift 2 ;;
        --queries) QUERIES_PER_SEED="$2"; shift 2 ;;
        *) echo "Unknown: $1"; exit 1 ;;
    esac
done

# State file tracks: ROUND_NUMBER:STATUS (submitted|queried|gt_fetched)
touch "$STATE"

log() {
    local msg="[$(date '+%Y-%m-%d %H:%M:%S')] $1"
    echo "$msg"
    echo "$msg" >> "$LOG"
}

round_state() {
    grep "^${1}:" "$STATE" 2>/dev/null | cut -d: -f2 || echo "new"
}

set_state() {
    # Remove old entry, add new
    grep -v "^${1}:" "$STATE" > "${STATE}.tmp" 2>/dev/null || true
    echo "${1}:${2}" >> "${STATE}.tmp"
    mv "${STATE}.tmp" "$STATE"
}

log "============================================"
log "  Watch Rounds — Started"
log "  Interval: ${INTERVAL_MIN} min"
log "  Rollouts: ${ROLLOUTS}"
log "  Queries: ${QUERIES_PER_SEED}/seed"
log "============================================"

while true; do
    log "── Checking for active round ──"

    # Query API
    STATUS_JSON=""
    if STATUS_JSON=$(cd aggregate-data && deno run -A check-round.ts 2>>"$LOG"); then
        ACTIVE_ROUND=$(echo "$STATUS_JSON" | python3 -c "import sys,json; d=json.load(sys.stdin); print(d['active_round'] or '')")
        QUERIES_USED=$(echo "$STATUS_JSON" | python3 -c "import sys,json; d=json.load(sys.stdin); print(d['queries_used'])")
        QUERIES_MAX=$(echo "$STATUS_JSON" | python3 -c "import sys,json; d=json.load(sys.stdin); print(d['queries_max'])")
        SEEDS_SUB=$(echo "$STATUS_JSON" | python3 -c "import sys,json; d=json.load(sys.stdin); print(d['seeds_submitted'])")
        MINS_LEFT=$(echo "$STATUS_JSON" | python3 -c "import sys,json; d=json.load(sys.stdin); print(d['minutes_remaining'] or 0)")
        COMPLETED=$(echo "$STATUS_JSON" | python3 -c "import sys,json; d=json.load(sys.stdin); print(','.join(map(str,d['completed_rounds'])))")
    else
        log "  API check failed, will retry next cycle"
        sleep $((INTERVAL_MIN * 60))
        continue
    fi

    # ── Handle active round ──
    if [ -n "$ACTIVE_ROUND" ]; then
        STATE_VAL=$(round_state "$ACTIVE_ROUND")
        log "  Active: Round $ACTIVE_ROUND | ${MINS_LEFT} min left | ${SEEDS_SUB}/5 seeds | ${QUERIES_USED}/${QUERIES_MAX} queries | state: $STATE_VAL"

        if [ "$STATE_VAL" = "new" ]; then
            # Brand new round — full pipeline
            log "  >>> NEW ROUND $ACTIVE_ROUND — running full pipeline"

            if ./prepare_for_round.sh "$ACTIVE_ROUND" \
                --rollouts "$ROLLOUTS" \
                --queries "$QUERIES_PER_SEED" \
                --viewport "$VIEWPORT" \
                --smart-alloc \
                >> "$LOG" 2>&1; then
                set_state "$ACTIVE_ROUND" "submitted"
                log "  >>> Round $ACTIVE_ROUND submitted successfully!"
            else
                log "  >>> ERROR: Pipeline failed for round $ACTIVE_ROUND"
                set_state "$ACTIVE_ROUND" "failed"
            fi

        elif [ "$STATE_VAL" = "submitted" ] || [ "$STATE_VAL" = "failed" ]; then
            # Already submitted — use remaining queries if any
            QUERIES_LEFT=$((QUERIES_MAX - QUERIES_USED))
            if [ "$QUERIES_LEFT" -gt 0 ] && [ "$MINS_LEFT" -gt 5 ]; then
                log "  >>> $QUERIES_LEFT queries remaining — running additional queries + re-submit"

                cd aggregate-data
                if deno run -A query-round.ts "$ACTIVE_ROUND" \
                    --queries-per-seed "$QUERIES_PER_SEED" \
                    --viewport "$VIEWPORT" \
                    --smart-alloc \
                    >> "../$LOG" 2>&1; then

                    deno run -A submit-predictions.ts --round "$ACTIVE_ROUND" >> "../$LOG" 2>&1
                    log "  >>> Re-submitted round $ACTIVE_ROUND with extra query data"
                else
                    log "  >>> Extra queries failed (non-fatal)"
                fi
                cd ..

                set_state "$ACTIVE_ROUND" "queried"
            else
                log "  >>> Already fully processed (${QUERIES_USED}/${QUERIES_MAX} queries used)"
            fi
        else
            log "  >>> Round $ACTIVE_ROUND already fully processed"
        fi
    else
        log "  No active round"
    fi

    # ── Fetch GT for completed rounds ──
    if [ -n "$COMPLETED" ]; then
        IFS=',' read -ra COMP_ROUNDS <<< "$COMPLETED"
        for R in "${COMP_ROUNDS[@]}"; do
            GT_STATE=$(round_state "gt_${R}")
            if [ "$GT_STATE" = "new" ]; then
                log "  Fetching ground truth for completed round $R"
                cd aggregate-data
                if deno run -A fetch-analysis.ts >> "../$LOG" 2>&1; then
                    deno run -A convert-to-bin.ts >> "../$LOG" 2>&1
                    set_state "gt_${R}" "fetched"
                    log "  GT for round $R fetched & converted"
                else
                    log "  GT fetch failed for round $R (will retry)"
                fi
                cd ..
                break  # one GT fetch per cycle is enough
            fi
        done
    fi

    # ── Sleep ──
    # If a round is active and closing soon, check more frequently
    if [ -n "$ACTIVE_ROUND" ] && [ "$MINS_LEFT" -gt 0 ] && [ "$MINS_LEFT" -lt 30 ]; then
        SLEEP_MIN=5
        log "  Round closing soon! Next check in ${SLEEP_MIN} min"
    elif [ -n "$ACTIVE_ROUND" ] && [ "$MINS_LEFT" -gt 0 ] && [ "$MINS_LEFT" -lt "$INTERVAL_MIN" ]; then
        SLEEP_MIN=$((MINS_LEFT / 2))
        [ "$SLEEP_MIN" -lt 10 ] && SLEEP_MIN=10
        log "  Next check in ${SLEEP_MIN} min (round closes in ${MINS_LEFT} min)"
    else
        SLEEP_MIN=$INTERVAL_MIN
        log "  Next check in ${SLEEP_MIN} min"
    fi

    sleep $((SLEEP_MIN * 60))
done
