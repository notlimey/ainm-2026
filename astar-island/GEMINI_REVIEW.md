# Astar Island — Strategy Review & Recommendations

## 1. Current Strategy Analysis

Based on the provided codebase (`predict.cpp`, `query-and-refine.ts`, and scripts), the current pipeline utilizes an **Empirical Bucket-Based Prior + Active Simulation Refinement** approach.

### How it works:
1. **Data Aggregation**: Past rounds' initial states and corresponding ground-truth probability distributions are fetched and converted into binary datasets.
2. **Empirical Modeling (C++)**: 
   - Extracts local, hand-crafted features for each cell (e.g., `terrain_class`, `is_coastal`, `adj_settlement`, `dist_nearest_settlement`).
   - Bins these features into discrete `BucketKey` and `BucketKeyCoarse` structures.
   - For a new map, it maps each cell's features to the corresponding bucket and uses the historical average ground-truth distribution for that bucket.
   - Falls back to coarse buckets if samples `< 10`, or a hardcoded fallback (`0.94` for current class) if no data exists.
   - **Crucial Safety**: Applies a probability floor (`0.01`) and renormalizes to avoid infinite KL divergence penalties.
3. **Active Refinement (TypeScript)**:
   - Calculates the entropy of the C++ model's predictions to find uncertain regions.
   - Selects high-entropy viewports and uses the round's limited budget (`/simulate` API) to query them.
   - Blends the empirical distribution of the query results with the C++ model's prior using a heuristic weight ($\alpha$).

### Strengths:
- **Extremely Fast and Robust**: The C++ bucket lookup is instantaneous and won't suffer from training instabilities or overfitting in the traditional ML sense.
- **Safety First**: Strict adherence to the `0.01` floor prevents disastrous `0` scores.
- **Smart Budget Allocation**: Viewport targeting based on predictive entropy maximizes the value of the 50 queries per round.
- **Data-Driven**: It correctly leverages the `/analysis` endpoint, which is the ultimate gold-standard training signal.

### Weaknesses:
- **Lossy Feature Engineering**: Discretizing continuous variables (like distance `dist_settle_bin`) loses valuable gradient/spatial information. Hand-crafted features might miss complex spatial interactions (e.g., a settlement protected by mountains vs. an exposed one).
- **Hardcoded Fallbacks**: The fallback behavior is naive and doesn't leverage similarities between states.
- **Heuristic Blending**: The $\alpha = \min(n / (n+5), 0.8)$ blending in `query-and-refine.ts` is arbitrary and may underweight strong model priors or overweight noisy sparse queries.

---

## 2. Recommendations for Improvement

To achieve a higher leaderboard ranking, the strategy needs to move beyond discrete lookup tables and start capturing complex spatial relationships. 

### Phase 1: Easy Wins (Modifications to Current System)
1. **Improve MLP Features**: The existing `mlp.cpp` normalizes handcrafted features. You can improve this by replacing discrete distance bins and adjacency counts with continuous, global distance fields (e.g., distance to nearest port, distance to nearest rival settlement) fed directly into the MLP.
2. **Bayesian Query Blending**: Update `query-and-refine.ts` to use a true Bayesian update. Treat the C++ model's (Bucket or MLP) output as a Dirichlet prior (parameterized by a pseudo-count of confidence) and the `/simulate` queries as multinomial observations. This will perfectly mathematically blend the two.
3. **Temporal Features**: The simulation runs for 50 years. Ensure features account for "momentum" (e.g., dense clusters of settlements grow faster than isolated ones).

### Phase 2: Move to Deep Learning (Spatial ML)
The current `mlp.cpp` evaluates each cell independently based on isolated features. The Astar Island simulation is essentially a complex Cellular Automaton. **Convolutional Neural Networks (CNNs)** are exceptionally well-suited for predicting CA outcomes because they inherently understand spatial locality and translation invariance.

**Proposed Architecture (U-Net or Fully Convolutional Network):**
- **Input (40x40xN)**: A multi-channel grid representing the initial state.
  - Channels for terrain types (One-hot encoded: Ocean, Plains, Mountain, Forest, Settlement, Port, Ruin).
  - Channels for global distance fields (Distance to coast, Distance to nearest settlement).
- **Network**: A lightweight U-Net or a stack of ResNet blocks.
  - *Why U-Net?* It captures both local interactions (who raids whom) and global context (are we on an isolated island or a crowded continent) and outputs a dense 40x40 grid.
- **Output (40x40x6)**: Softmax activation over the 6 prediction classes for every cell.
- **Loss Function**: Weighted KL Divergence (exactly matching the competition's scoring metric). Use the entropy of the ground truth cells to weight the loss during training, so the network focuses entirely on dynamic cells.

### Phase 3: The "Local Simulator" Approach (Advanced)
Since the competition gives you the initial state and the simulation is deterministic aside from a `sim_seed`, the ultimate strategy is to build a fast, approximate local emulator.
- If you can approximate the game rules (Growth, Conflict, Trade, Winter, Environment) in C++, you can run 1,000 Monte Carlo rollouts locally in seconds without using *any* API budget.
- Your API budget can then be spent solely on verifying your emulator's assumptions rather than generating the primary prediction.

## 3. Action Plan

If we want to iterate right now, I suggest we take the following path:
1. **Short Term**: The `mlp.cpp` model is a great starting point. We should optimize its hyper-parameters (epochs, learning rate, architecture) and experiment with feeding it better continuous features instead of relying entirely on the discretely binned Bucket approach. Furthermore, adding an ensemble method that combines the bucket lookup and the MLP predictions will likely boost the score.
2. **Medium Term**: Start building a PyTorch dataset from `grids.bin` and `ground_truth.bin` to train a lightweight CNN.
