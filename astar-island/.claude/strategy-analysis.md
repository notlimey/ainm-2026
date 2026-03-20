# Astar Island — Strategy Analysis

## Task Summary
Predict a 40x40x6 probability tensor per seed (5 seeds per round). Each cell's 6 values represent P(terrain_class) after 50 simulated years. Scored by entropy-weighted KL divergence — only dynamic cells matter.

## Terrain Classes (prediction)
| Class | Terrains mapped |
|-------|----------------|
| 0 | Ocean (10), Plains (11), Empty (0) |
| 1 | Settlement |
| 2 | Port |
| 3 | Ruin |
| 4 | Forest |
| 5 | Mountain |

## Scoring
- Ground truth = organizers run simulation hundreds of times → per-cell probability distribution
- KL(truth || pred) per cell, weighted by cell entropy
- `score = max(0, min(100, 100 * exp(-3 * weighted_kl)))`
- **NEVER predict 0.0** — clamp to 0.01 floor, renormalize

## Key API Endpoints
| Endpoint | Cost | Value |
|----------|------|-------|
| GET /rounds/{id} | Free | Initial states for all 5 seeds |
| POST /simulate | 1 query (50/round) | One stochastic run, 15x15 viewport |
| GET /analysis/{round_id}/{seed} | Free (post-round) | **Ground truth distributions** |
| GET /my-rounds | Free | Scores + completed round list |
| GET /my-predictions/{round_id} | Free | Past submissions |

## Critical Insight: /analysis is the training signal
After a round completes, /analysis returns the actual ground truth probability distribution AND your prediction. Pairing initial_state → ground_truth gives perfect training data for any model approach.

## Strategy: Hybrid approach
1. **Rules layer**: Ocean→0, Mountain→5, cells far from settlements→initial terrain
2. **Empirical layer**: From /analysis ground truth, learn P(class | local features) for dynamic cells
3. **Optional ML**: Per-cell classifier trained on ground truth, features = neighborhood, distance to settlement, etc.

## Simulation Mechanics (5 phases per year)
1. Growth — settlements produce food, grow population, found new settlements nearby
2. Conflict — raiding between settlements, longships extend range
3. Trade — ports trade if not at war, generates wealth
4. Winter — food loss, settlements can collapse → Ruins
5. Environment — ruins reclaimed by nearby settlements or overtaken by forest

## Budget Strategy
- 50 queries/round across 5 seeds (10 per seed if even split)
- Each /simulate is a different stochastic outcome
- Focus queries on areas near settlements (high entropy cells)
- /analysis ground truth is better than simulate for model training

## Data Collection Priorities
1. P0: Fetch /analysis for ALL completed rounds/seeds
2. P0: Fetch /my-rounds to identify completed rounds
3. P1: Build training dataset: initial_state → ground_truth pairs
4. P1: Scrape more seeds for trajectory understanding
5. P2: Statistical analysis of per-cell features vs outcome
6. P3: Train per-cell classifier
