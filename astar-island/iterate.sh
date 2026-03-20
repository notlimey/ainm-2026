#!/bin/bash
set -e

ROUND=${1:?Usage: ./iterate.sh <round_number> [mlp_epochs] [model]}
EPOCHS=${2:-7}
MODEL=${3:-ensemble}  # bucket, mlp, ensemble

echo "=========================================="
echo "  Iterate Round $ROUND"
echo "  Epochs: $EPOCHS  Model: $MODEL"
echo "=========================================="

# 1. Fetch + convert (only if --refresh passed)
if [[ "$*" == *"--refresh"* ]]; then
    echo ""
    echo "=== Refreshing data ==="
    cd aggregate-data
    deno run -A fetch-analysis.ts
    deno run -A convert-to-bin.ts
    cd ..
fi

cd simulation

# 2. Build if needed
[ -f predict ] || c++ -std=c++17 -O2 -o predict predict.cpp
[ -f mlp ]     || c++ -std=c++17 -O2 -o mlp mlp.cpp

# 3. Generate bucket predictions (saved to pred_bucket_r{N}_s{S}.bin to avoid overwrite)
echo ""
echo "=== Bucket predictions ==="
for S in 0 1 2 3 4; do
    ./predict data/training.bin data/grids.bin "$ROUND" "$S" "data/pred_bucket_r${ROUND}_s${S}.bin" --exclude "$ROUND" 2>&1 | grep -E "Score|Written|Validation" || true
    # Also copy to pred_r{N}_s{S}.bin as default submission file
    cp "data/pred_bucket_r${ROUND}_s${S}.bin" "data/pred_r${ROUND}_s${S}.bin"
done

echo ""
echo "=== MLP predictions (${EPOCHS} epochs) ==="
for S in 0 1 2 3 4; do
    ./mlp data/training.bin data/grids.bin "$ROUND" "$S" "data/pred_mlp_r${ROUND}_s${S}.bin" \
        --exclude "$ROUND" --epochs "$EPOCHS" 2>&1 | grep -E "Score|Written|Validation" || true
done

# 4. Blend with stored queries if any exist
echo ""
echo "=== Blending (model: $MODEL) ==="
cd ../aggregate-data
deno run -A query-round.ts "$ROUND" --blend-only --model "$MODEL" 2>&1 | grep -E "Loaded|Stored|Blended|Written|Submit|Seed|ensemble" || true

# 5. Generate scores
echo ""
echo "=== Scores ==="
deno run -A preview-prediction.ts 2>&1 | grep -E "^S[0-9]|^Seed|^-|R[0-9]|AVG|Scores CSV"

echo ""
echo "Happy? Submit with:"
echo "  cd aggregate-data && deno run -A submit-predictions.ts --round $ROUND"
