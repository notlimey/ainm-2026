#!/bin/bash
set -e

# Sweep MLP epochs to find optimal training duration.
# Tests each epoch count across all available rounds (leave-one-out)
# and reports average score per epoch setting.

EPOCHS_LIST="3 5 7 10 15 20 30 50 80 120"

# Build MLP if needed
[ -f mlp ] && [ mlp -nt mlp.cpp ] || { echo "Compiling mlp..."; c++ -std=c++17 -O3 -o mlp mlp.cpp; }

# Discover available rounds
ROUNDS=""
[ -f predict ] && [ predict -nt predict.cpp ] || { echo "Compiling predict..."; c++ -std=c++17 -O3 -o predict predict.cpp; }
for R in $(seq 1 15); do
    if ./predict data/training.bin data/grids.bin "$R" 0 /dev/null --exclude "$R" 2>&1 | grep -q "Grid \["; then
        ROUNDS="$ROUNDS $R"
    fi
done
ROUNDS=$(echo $ROUNDS | xargs)
echo "Rounds: $ROUNDS"
echo ""

# Header
printf "%-8s" "Epochs"
for R in $ROUNDS; do
    printf "  R%-5s" "$R"
done
printf "  %-8s\n" "AVG"
printf "%-8s" "------"
for R in $ROUNDS; do
    printf "  ------"
done
printf "  --------\n"

# Run sweep
for EP in $EPOCHS_LIST; do
    TOTAL=0
    COUNT=0
    LINE=$(printf "%-8s" "$EP")

    for R in $ROUNDS; do
        # Just use seed 0 for speed (representative enough)
        SCORE=$(./mlp data/training.bin data/grids.bin "$R" 0 /dev/null --exclude "$R" --epochs "$EP" 2>&1 | grep "Score:" | awk '{print $2}')
        if [ -n "$SCORE" ]; then
            LINE="$LINE  $(printf "%6s" "$SCORE")"
            TOTAL=$(echo "$TOTAL + $SCORE" | bc)
            COUNT=$((COUNT + 1))
        else
            LINE="$LINE  $(printf "%6s" "N/A")"
        fi
    done

    if [ "$COUNT" -gt 0 ]; then
        AVG=$(echo "scale=2; $TOTAL / $COUNT" | bc)
        LINE="$LINE  $(printf "%8s" "$AVG")"
    else
        LINE="$LINE  $(printf "%8s" "N/A")"
    fi

    echo "$LINE"
done
