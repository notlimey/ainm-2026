# Astar Island — Norse World Prediction Challenge

## Overview

Astar Island is a challenge from the Norwegian AI Championship (NM i AI). You observe a procedurally generated Norse world simulation — a grid of terrain, settlements, ports, and ruins — then predict what the world will look like after 50 simulated years.

Your goal: submit a **probability distribution** over terrain classes for every cell on the map that closely matches the ground truth distribution (computed from hundreds of Monte Carlo runs by the organizers).

---

## The World

The world is a **40×40 rectangular grid** with 8 internal terrain types that map to **6 prediction classes**:

| Internal Code | Terrain    | Class Index | Description                          |
|---------------|------------|-------------|--------------------------------------|
| 10            | Ocean      | 0 (Empty)   | Impassable water, borders the map    |
| 11            | Plains     | 0 (Empty)   | Flat land, buildable                 |
| 0             | Empty      | 0           | Generic empty cell                   |
| 1             | Settlement | 1           | Active Norse settlement              |
| 2             | Port       | 2           | Coastal settlement with harbour      |
| 3             | Ruin       | 3           | Collapsed settlement                 |
| 4             | Forest     | 4           | Provides food to adjacent settlements|
| 5             | Mountain   | 5           | Impassable terrain                   |

- Ocean, Plains, and Empty all map to **class 0** in predictions.
- Mountains are **static** (never change).
- Forests are mostly static but can reclaim ruined land.
- The interesting cells are those that can become Settlements, Ports, or Ruins.

### Map Generation

Each map is procedurally generated from a **map seed**:

- **Ocean borders** surround the map
- **Fjords** cut inland from random edges
- **Mountain chains** form via random walks
- **Forest patches** cover land with clustered groves
- **Initial settlements** placed on land cells, spaced apart

The map seed is visible to you — you can reconstruct the initial terrain layout locally.

---

## Simulation Lifecycle

Each of the **50 simulated years** cycles through multiple phases. The world goes through **growth, conflict, trade, harsh winters, and environmental change** — in that order.

### 1. Growth

Settlements produce food based on adjacent terrain. When conditions are right, settlements grow in population, develop ports along coastlines, and build longships for naval operations. Prosperous settlements expand by founding new settlements on nearby land.

### 2. Conflict

Settlements raid each other. Longships extend raiding range significantly. Desperate settlements (low food) raid more aggressively. Successful raids loot resources and damage the defender. Sometimes, conquered settlements change allegiance to the raiding faction.

### 3. Trade

Ports within range of each other can trade if not at war. Trade generates wealth and food for both parties, and technology diffuses between trading partners.

### 4. Winter

Each year ends with a winter of varying severity. All settlements lose food. Settlements can collapse from starvation, sustained raids, or harsh winters — becoming **Ruins** and dispersing population to nearby friendly settlements.

### 5. Environment

The natural world slowly reclaims abandoned land. Nearby thriving settlements may reclaim and rebuild ruined sites, establishing new outposts that inherit a portion of their patron's resources and knowledge. Coastal ruins can even be restored as ports. If no settlement steps in, ruins are eventually overtaken by forest growth or fade back into open plains.

### Settlement Properties

Each settlement tracks: position, population, food, wealth, defense, tech level, port status, longship ownership, and faction allegiance (owner_id).

Initial states expose settlement positions and port status. Internal stats (population, food, wealth, defense) are only visible through simulation queries.

---

## Rounds & Seeds

- Each **round** contains **5 seeds** (map configurations).
- You have a **prediction window** (typically ~2 hours 45 minutes) to query and submit.
- You must submit a prediction for each seed. Missing seeds score **0**.

### Round Lifecycle

| Status      | Meaning                                 |
|-------------|-----------------------------------------|
| `pending`   | Round created but not yet started       |
| `active`    | Queries and submissions open            |
| `scoring`   | Submissions closed, scoring in progress |
| `completed` | Scores finalized                        |

---

## API Endpoints

**Base URL:** `https://api.ainm.no/astar-island`

All endpoints require authentication via JWT token (cookie `access_token` or `Authorization: Bearer <token>` header).

### Key Endpoints

| Method | Path                                      | Auth   | Description                                |
|--------|-------------------------------------------|--------|--------------------------------------------|
| `GET`  | `/rounds`                                 | Public | List all rounds                            |
| `GET`  | `/rounds/{round_id}`                      | Public | Round details + initial states             |
| `GET`  | `/budget`                                 | Team   | Query budget for active round              |
| `POST` | `/simulate`                               | Team   | Observe one simulation through viewport    |
| `POST` | `/submit`                                 | Team   | Submit prediction tensor                   |
| `GET`  | `/my-rounds`                              | Team   | Rounds with your scores, rank, budget      |
| `GET`  | `/my-predictions/{round_id}`              | Team   | Your predictions with argmax/confidence    |
| `GET`  | `/analysis/{round_id}/{seed_index}`       | Team   | Post-round ground truth comparison         |
| `GET`  | `/leaderboard`                            | Public | Astar Island leaderboard                   |

### Rate Limits

| Endpoint          | Limit                      |
|-------------------|----------------------------|
| `POST /simulate`  | 5 requests/second per team |
| `POST /submit`    | 2 requests/second per team |

### Simulation Queries (`POST /simulate`)

This is the **core observation endpoint**. Each call runs one stochastic simulation and reveals a viewport window of the result. Costs **one query** from your budget (**50 per round**, shared across all 5 seeds).

**Request:**

```json
{
  "round_id": "uuid-of-active-round",
  "seed_index": 3,
  "viewport_x": 10,
  "viewport_y": 5,
  "viewport_w": 15,
  "viewport_h": 15
}
```

- `seed_index`: 0–4 (which of the 5 seeds to simulate)
- `viewport_x/y`: top-left corner of viewport (clamped to map edges)
- `viewport_w/h`: viewport dimensions, range **5–15** cells each

**Response:**

```json
{
  "grid": [[4, 11, 1, ...], ...],
  "settlements": [
    {
      "x": 12, "y": 7,
      "population": 2.8, "food": 0.4, "wealth": 0.7, "defense": 0.6,
      "has_port": true, "alive": true, "owner_id": 3
    }
  ],
  "viewport": {"x": 10, "y": 5, "w": 15, "h": 15},
  "width": 40, "height": 40,
  "queries_used": 24, "queries_max": 50
}
```

- The `grid` contains **only the viewport region**, not the full map.
- Each call uses a **different random sim_seed**, so you get a different stochastic outcome each time.

### Submitting Predictions (`POST /submit`)

For each seed, submit an **H×W×6 probability tensor**:

```json
{
  "round_id": "uuid-of-active-round",
  "seed_index": 3,
  "prediction": [
    [
      [0.85, 0.05, 0.02, 0.03, 0.03, 0.02],
      [0.10, 0.40, 0.30, 0.10, 0.05, 0.05]
    ]
  ]
}
```

- `prediction[y][x][class]` — probability per cell per class
- Each cell's 6 probabilities must sum to **1.0** (±0.01 tolerance)
- All probabilities must be non-negative
- Resubmitting overwrites the previous prediction for that seed

**Class Indices:**

| Index | Class                          |
|-------|--------------------------------|
| 0     | Empty (Ocean, Plains, Empty)   |
| 1     | Settlement                     |
| 2     | Port                           |
| 3     | Ruin                           |
| 4     | Forest                         |
| 5     | Mountain                       |

### Post-Round Analysis (`GET /analysis/{round_id}/{seed_index}`)

Available after a round completes. Returns your prediction alongside the ground truth probability tensor for detailed comparison.

---

## Scoring

### Ground Truth

For each seed, organizers run the simulation **hundreds of times** with the true hidden parameters. This produces a probability distribution for each cell. For example, a cell might have ground truth `[0.0, 0.60, 0.25, 0.15, 0.0, 0.0]` — meaning 60% Settlement, 25% Port, 15% Ruin.

### KL Divergence

For each cell, [KL divergence](https://en.wikipedia.org/wiki/Kullback%E2%80%93Leibler_divergence) measures how different your prediction is from the ground truth:

```
KL(p || q) = Σ pᵢ × log(pᵢ / qᵢ)
```

Where `p` = ground truth, `q` = your prediction. Lower KL = better match.

### Entropy Weighting

Not all cells matter equally. **Static cells** (ocean, mountain) have near-zero entropy and are excluded. **Only dynamic cells** (those that change between simulation runs) contribute, weighted by their entropy:

```
entropy(cell) = -Σ pᵢ × log(pᵢ)
```

Cells with higher entropy (more uncertain outcomes) count more toward your score.

### Final Score Formula

```
weighted_kl = Σ entropy(cell) × KL(ground_truth[cell], prediction[cell])
              ─────────────────────────────────────────────────────────
                            Σ entropy(cell)

score = max(0, min(100, 100 × exp(-3 × weighted_kl)))
```

- **100** = perfect prediction (matches ground truth exactly)
- **0** = terrible prediction (high KL divergence)

### Per-Round & Leaderboard Scoring

- **Round score** = average of 5 per-seed scores
- **Leaderboard score** = your **best round score of all time** (weighted by round weight)
- A **hot streak score** (average of last 3 rounds) is also tracked
- Missing seed submissions score **0** — always submit something

---

## Critical Pitfalls

### Never Assign Probability 0.0

If the ground truth has `pᵢ > 0` but your prediction has `qᵢ = 0`, KL divergence goes to **infinity** — destroying your entire score for that cell.

**Always enforce a minimum probability floor:**

```python
prediction = np.maximum(prediction, 0.01)
prediction = prediction / prediction.sum(axis=-1, keepdims=True)
```

### Always Submit All 5 Seeds

Missing seeds score 0. Even a **uniform prediction** (`1/6` per class) beats not submitting at all (scores ~1–5).

---

## Quickstart

### Authentication

Log in at `app.ainm.no`, grab your `access_token` JWT from cookies.

```python
import requests

BASE = "https://api.ainm.no"
session = requests.Session()
session.headers["Authorization"] = "Bearer YOUR_JWT_TOKEN"
```

### Step 1: Get the Active Round

```python
rounds = session.get(f"{BASE}/astar-island/rounds").json()
active = next((r for r in rounds if r["status"] == "active"), None)
round_id = active["id"]
```

### Step 2: Get Round Details & Initial States

```python
detail = session.get(f"{BASE}/astar-island/rounds/{round_id}").json()

width = detail["map_width"]      # 40
height = detail["map_height"]    # 40
seeds = detail["seeds_count"]    # 5

for i, state in enumerate(detail["initial_states"]):
    grid = state["grid"]                    # height × width terrain codes
    settlements = state["settlements"]      # [{x, y, has_port, alive}, ...]
```

### Step 3: Query the Simulator (50 queries total)

```python
result = session.post(f"{BASE}/astar-island/simulate", json={
    "round_id": round_id,
    "seed_index": 0,
    "viewport_x": 10,
    "viewport_y": 5,
    "viewport_w": 15,
    "viewport_h": 15,
}).json()
```

### Step 4: Build & Submit Predictions

```python
import numpy as np

for seed_idx in range(seeds):
    prediction = np.full((height, width, 6), 1/6)  # uniform baseline

    # TODO: replace with your model's predictions

    # Safety floor — never assign 0 probability
    prediction = np.maximum(prediction, 0.01)
    prediction = prediction / prediction.sum(axis=-1, keepdims=True)

    resp = session.post(f"{BASE}/astar-island/submit", json={
        "round_id": round_id,
        "seed_index": seed_idx,
        "prediction": prediction.tolist(),
    })
    print(f"Seed {seed_idx}: {resp.status_code}")
```

---

## Strategy Tips

1. **Static cells are free points** — Ocean, Mountain, and Forest cells rarely change. Assign high confidence to their current class.
2. **Focus queries on dynamic areas** — Use your 50 queries to observe areas near settlements where growth, conflict, and collapse happen.
3. **Aggregate multiple simulations** — Each query gives a different stochastic outcome. Run multiple queries on the same viewport and average the results to estimate the probability distribution.
4. **Viewport placement matters** — Max viewport is 15×15. Place viewports to cover settlement clusters for maximum information.
5. **Run your own simulation** — The map seed is visible. Reconstruct the world locally and run your own Monte Carlo simulations to estimate distributions without spending API queries.
6. **Probability floor is essential** — Always clamp to ≥0.01 and renormalize. A single 0.0 where ground truth is nonzero destroys your score.
