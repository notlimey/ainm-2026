#!/bin/bash
# Benchmark MLP across rounds, seeds, and epoch counts
# Usage: ./benchmark_mlp.sh [epoch1 epoch2 ...] (default: 10 50 100 200)

set -e

EPOCHS=(${@:-10 50 100 200})
ROUNDS=(1 2 3 4 5 6)
SEEDS=(0 1 2 3 4)

TRAINING="data/training.bin"
GRIDS="data/grids.bin"
TMP_PRED="/tmp/mlp_bench_pred.bin"

echo "============================================="
echo "  MLP Benchmark"
echo "  Rounds: ${ROUNDS[*]}"
echo "  Seeds:  ${SEEDS[*]}"
echo "  Epochs: ${EPOCHS[*]}"
echo "============================================="
echo ""

# Header
printf "%-8s" "Epochs"
for r in "${ROUNDS[@]}"; do
    printf "%-12s" "R${r}"
done
printf "%-12s\n" "AVG"
printf '%0.s-' {1..88}
echo ""

for ep in "${EPOCHS[@]}"; do
    total_score=0
    count=0
    round_avgs=()

    for r in "${ROUNDS[@]}"; do
        round_total=0
        round_count=0

        for s in "${SEEDS[@]}"; do
            # Run MLP: exclude the target round, capture score
            output=$(./mlp "$TRAINING" "$GRIDS" "$r" "$s" "$TMP_PRED" \
                --exclude "$r" --epochs "$ep" 2>&1)

            score=$(echo "$output" | grep "Score:" | awk '{print $2}')
            if [ -n "$score" ]; then
                round_total=$(echo "$round_total + $score" | bc)
                total_score=$(echo "$total_score + $score" | bc)
                round_count=$((round_count + 1))
                count=$((count + 1))
            fi
        done

        if [ "$round_count" -gt 0 ]; then
            round_avg=$(echo "scale=2; $round_total / $round_count" | bc)
        else
            round_avg="N/A"
        fi
        round_avgs+=("$round_avg")
    done

    # Overall average
    if [ "$count" -gt 0 ]; then
        overall_avg=$(echo "scale=2; $total_score / $count" | bc)
    else
        overall_avg="N/A"
    fi

    # Print row
    printf "%-8s" "$ep"
    for avg in "${round_avgs[@]}"; do
        printf "%-12s" "$avg"
    done
    printf "%-12s\n" "$overall_avg"
done

echo ""
echo "Done. Each cell = average score across seeds ${SEEDS[*]} (--exclude round)."

rm -f "$TMP_PRED"
