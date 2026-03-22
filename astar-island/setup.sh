#!/bin/bash
# One-time setup: compile all C++ binaries, check dependencies, create dirs.
# Run from project root.
set -e
cd "$(dirname "$0")"

echo "=========================================="
echo "  Astar Island — Setup"
echo "=========================================="

# Detect platform
if [[ "$(uname)" == "Linux" ]]; then
    CXX="${CXX:-g++}"
    EXTRA="-pthread"
    echo "  Platform: Linux (using $CXX, -pthread)"
else
    CXX="${CXX:-c++}"
    EXTRA=""
    echo "  Platform: macOS (using $CXX)"
fi

# Check deno
if ! command -v deno &>/dev/null; then
    echo "  ERROR: deno not found. Install: https://deno.land"
    exit 1
fi
echo "  Deno: $(deno --version | head -1)"

# Create data dirs
mkdir -p simulation/data
mkdir -p aggregate-data/data/queries
echo "  Data dirs: OK"

# Compile all C++ binaries
echo ""
echo "=== Compiling C++ ==="
cd simulation

compile() {
    local src="$1"
    local bin="${src%.cpp}"
    if [ ! -f "$bin" ] || [ "$src" -nt "$bin" ] || [ "$FORCE" = "1" ]; then
        echo "  Compiling $src..."
        $CXX -std=c++17 -O3 $EXTRA -o "$bin" "$src"
    else
        echo "  $bin: up to date"
    fi
}

compile simulate.cpp
compile calibrate.cpp
compile cnn.cpp
compile predict.cpp
compile mlp.cpp
compile blend.cpp

cd ..

# Check for calibrated params
echo ""
echo "=== Checking data files ==="
[ -f simulation/data/params.bin ] && echo "  params.bin: found" || echo "  params.bin: MISSING (run calibrate on Ryzen)"
[ -f simulation/data/grids.bin ] && echo "  grids.bin: found" || echo "  grids.bin: MISSING (run fetch-analysis.ts + convert-to-bin.ts)"
[ -f simulation/data/ground_truth.bin ] && echo "  ground_truth.bin: found" || echo "  ground_truth.bin: MISSING (run fetch-analysis.ts + convert-to-bin.ts)"

echo ""
echo "Setup complete. Run: ./prepare_for_round.sh <ROUND>"
