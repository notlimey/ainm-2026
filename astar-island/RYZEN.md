# Ryzen Instructions

Step-by-step instructions to run on the Ryzen 9 5950X / RTX 3090 Ti (Linux, fish shell).

## 1. Pull latest code

```fish
cd ~/astar-island   # or wherever the repo lives
git pull
```

## 2. Compile everything

```fish
cd simulation
g++ -std=c++17 -O3 -pthread -o simulate simulate.cpp
g++ -std=c++17 -O3 -pthread -o calibrate calibrate.cpp
g++ -std=c++17 -O3 -pthread -o cnn cnn.cpp
g++ -std=c++17 -O3 -pthread -o tune tune.cpp
g++ -std=c++17 -O3 -o predict predict.cpp
g++ -std=c++17 -O3 -o mlp mlp.cpp
g++ -std=c++17 -O3 -o blend blend.cpp
```

## 3. Recalibrate global params (MOST IMPORTANT)

This is the #1 priority. We have 21 rounds of ground truth and a trade mechanics fix.
The old params.bin was calibrated on fewer rounds with the old trade code.

**Estimated time: ~60-90 min on 32 threads.**

```fish
cd simulation
./calibrate data/grids.bin data/ground_truth.bin \
  --generations 300 --rollouts 50 --threads 32 \
  --seeds-per-round 3 --output data/params_v5.bin
```

### Verify the new params are better

Quick comparison on 5 representative rounds (good + bad):

```fish
echo "=== Testing params_v5 vs params.bin ==="
for R in 9 15 17 18 20
  echo "--- Round $R ---"
  echo -n "  old: "; ./simulate data/grids.bin $R 0 /dev/null --rollouts 200 --params data/params.bin 2>&1 | grep Score
  echo -n "  new: "; ./simulate data/grids.bin $R 0 /dev/null --rollouts 200 --params data/params_v5.bin 2>&1 | grep Score
end
```

If v5 is better overall, replace the main params:
```fish
cp data/params.bin data/params_old_backup.bin
cp data/params_v5.bin data/params.bin
```

If worse, try with more generations:
```fish
./calibrate data/grids.bin data/ground_truth.bin \
  --generations 500 --rollouts 80 --threads 32 \
  --seeds-per-round 5 --output data/params_v5b.bin
```

## 4. Regenerate ALL sim predictions with new params

```fish
for R in (seq 1 22)
  for S in 0 1 2 3 4
    ./simulate data/grids.bin $R $S data/pred_sim_r{$R}_s{$S}.bin \
      --rollouts 2000 --params data/params.bin
  end
end
```

**Estimated time: ~5-10 min for 110 predictions.**

## 5. Generate CNN predictions (all rounds)

CNN needs sim predictions as input (reads `pred_sim_r{R}_s{S}.bin`).

```fish
for R in (seq 1 22)
  for S in 0 1 2 3 4
    ./cnn data/grids.bin data/ground_truth.bin $R $S data/pred_cnn_r{$R}_s{$S}.bin \
      --exclude $R --epochs 30 --sim-dir data --threads 32 --channels 32
  end
end
```

**Estimated time: ~20-30 min for 110 predictions.**

## 6. Generate bucket + MLP predictions

```fish
for R in (seq 1 22)
  for S in 0 1 2 3 4
    ./predict data/training.bin data/grids.bin $R $S data/pred_bucket_r{$R}_s{$S}.bin --exclude $R --epochs 7
    ./mlp data/training.bin data/grids.bin $R $S data/pred_mlp_r{$R}_s{$S}.bin --exclude $R --epochs 15
  end
end
```

## 7. Run blend optimizer

```fish
./blend data/ground_truth.bin --data-dir data/ --grids data/grids.bin
```

This shows the CV score and optimal per-class weights. Compare with previous CV (69.47) to measure improvement.

## 8. Push results back

```fish
cd ..
git add simulation/data/params.bin simulation/data/params_v5.bin
git commit -m "Ryzen: recalibrated params with trade fix + 21 rounds"
git push
```

The prediction files (.bin) are likely gitignored. Copy critical ones if needed:
```fish
# Only if we need them on Mac:
scp simulation/data/params_v5.bin mac:~/astar-island/simulation/data/
```

## What changed (for context)

1. **Trade mechanics fix** — simulate.cpp and calibrate.cpp now track war state per year. Factions that raided each other can't trade that year, but all other factions CAN trade (was: only same-faction trade). This should give settlements more food via cross-faction trade.

2. **21 rounds of GT** — ground_truth.bin now has 105 entries (21 rounds × 5 seeds). Previous calibration used fewer rounds.

3. **tune.cpp** — New per-round parameter tuning tool. Not needed for Ryzen batch jobs but compiled for completeness.

## Troubleshooting

- If `calibrate` segfaults: check that `data/grids.bin` and `data/ground_truth.bin` are up to date. Run `cd ../aggregate-data && deno run -A convert-to-bin.ts` to regenerate.
- If scores look worse: the trade fix changes the sim dynamics significantly. The calibrator needs enough generations (300+) to find the new optimum.
- If `cnn` errors about missing sim predictions: run step 4 first (sim predictions must exist before CNN can use them).
