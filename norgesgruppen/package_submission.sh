#!/bin/bash
# Package submission zip from trained model weights.
# Usage: ./package_submission.sh path/to/best.pt
set -e

MODEL_PATH="${1:?Usage: ./package_submission.sh path/to/best.pt}"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
SUBMISSION_DIR="$SCRIPT_DIR/submission"
OUTPUT="$SCRIPT_DIR/submission.zip"

# Clean and create submission dir
rm -rf "$SUBMISSION_DIR" "$OUTPUT"
mkdir -p "$SUBMISSION_DIR"

# Copy files
cp "$SCRIPT_DIR/run.py" "$SUBMISSION_DIR/"
cp "$MODEL_PATH" "$SUBMISSION_DIR/best.pt"

# Create zip
cd "$SUBMISSION_DIR"
zip -r "$OUTPUT" . -x ".*" "__MACOSX/*"

# Verify
echo ""
echo "=== Zip contents ==="
unzip -l "$OUTPUT" | head -10
echo ""
echo "=== Size ==="
du -h "$OUTPUT"
echo ""
echo "Ready to upload: $OUTPUT"
