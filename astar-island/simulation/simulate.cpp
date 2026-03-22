// simulate.cpp — Monte Carlo Norse World Simulator
// Runs N rollouts of a 50-year simulation, outputs probability distributions per cell.
// Usage: ./simulate data/grids.bin <round> <seed> [output.bin] [--rollouts N] [--params file]

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <vector>
#include <string>
#include <fstream>
#include <algorithm>
#include <random>
#include <numeric>
#include <set>
#include "database.hpp"
#include "features.hpp"
#include "sim_params.hpp"
#include "io.hpp"

// ─── Simulation Settlement ──────────────────────────────────────────────────

struct SimSettlement {
    int x, y;
    float population;
    float food;
    float wealth;
    float defense;
    float tech;
    bool has_port;
    bool alive;
    int owner_id;
    int longships;
};

// ─── Simulation World ───────────────────────────────────────────────────────

struct SimWorld {
    int W, H;
    std::vector<std::vector<int>> grid;
    std::vector<SimSettlement> settlements;
    std::mt19937 rng;

    // Lookup: grid position → settlement index (-1 if none)
    std::vector<std::vector<int>> settle_map;

    // War state: faction pairs that raided each other this year (cleared each year)
    // Stored as (min_id, max_id) pairs for canonical ordering
    std::set<std::pair<int,int>> war_pairs;

    void mark_war(int faction_a, int faction_b) {
        if (faction_a == faction_b) return;
        int lo = std::min(faction_a, faction_b);
        int hi = std::max(faction_a, faction_b);
        war_pairs.insert({lo, hi});
    }

    bool at_war(int faction_a, int faction_b) const {
        if (faction_a == faction_b) return false;
        int lo = std::min(faction_a, faction_b);
        int hi = std::max(faction_a, faction_b);
        return war_pairs.count({lo, hi}) > 0;
    }

    void rebuild_settle_map() {
        settle_map.assign(H, std::vector<int>(W, -1));
        for (int i = 0; i < (int)settlements.size(); i++) {
            auto& s = settlements[i];
            if (s.alive)
                settle_map[s.y][s.x] = i;
        }
    }

    bool is_land(int x, int y) const {
        if (x < 0 || x >= W || y < 0 || y >= H) return false;
        int t = grid[y][x];
        return t != TERRAIN_OCEAN;
    }

    bool is_coastal(int x, int y) const {
        if (!is_land(x, y)) return false;
        for (int d = 0; d < 8; d++) {
            int nx = x + DX[d], ny = y + DY[d];
            if (nx < 0 || nx >= W || ny < 0 || ny >= H) continue;
            if (grid[ny][nx] == TERRAIN_OCEAN) return true;
        }
        return false;
    }

    int count_adjacent(int x, int y, int terrain) const {
        int c = 0;
        for (int d = 0; d < 8; d++) {
            int nx = x + DX[d], ny = y + DY[d];
            if (nx < 0 || nx >= W || ny < 0 || ny >= H) continue;
            if (grid[ny][nx] == terrain) c++;
        }
        return c;
    }

    float randf() { return std::uniform_real_distribution<float>(0.0f, 1.0f)(rng); }
    float randn(float mean, float stddev) { return std::normal_distribution<float>(mean, stddev)(rng); }
};

// ─── Initialize world from grid ─────────────────────────────────────────────

SimWorld init_world(const std::vector<std::vector<int>>& initial_grid,
                    const SimParams& p, uint32_t seed) {
    SimWorld w;
    w.H = initial_grid.size();
    w.W = w.H > 0 ? initial_grid[0].size() : 0;
    w.rng.seed(seed);

    // Deep copy grid
    w.grid = initial_grid;
    w.settlements.reserve(w.W * w.H); // prevent reallocation invalidating references

    // Find settlements
    int next_owner = 0;
    for (int y = 0; y < w.H; y++) {
        for (int x = 0; x < w.W; x++) {
            int t = w.grid[y][x];
            if (t == TERRAIN_SETTLEMENT || t == TERRAIN_PORT) {
                SimSettlement s;
                s.x = x; s.y = y;
                s.population = p.init_population * (0.8f + w.randf() * 0.4f); // slight variation
                s.food = p.init_food * (0.8f + w.randf() * 0.4f);
                s.wealth = 0.0f;
                s.defense = p.init_defense * (0.8f + w.randf() * 0.4f);
                s.tech = p.init_tech;
                s.has_port = (t == TERRAIN_PORT);
                s.alive = true;
                s.owner_id = next_owner++;
                s.longships = s.has_port ? 1 : 0;
                w.settlements.push_back(s);
            }
        }
    }

    w.rebuild_settle_map();
    return w;
}

// ─── Phase 1: Growth ────────────────────────────────────────────────────────

void phase_growth(SimWorld& w, const SimParams& p) {
    // Collect expansion candidates first to avoid modifying during iteration
    struct ExpansionCandidate {
        int parent_idx;
        int tx, ty;
    };
    std::vector<ExpansionCandidate> expansions;

    for (int i = 0; i < (int)w.settlements.size(); i++) {
        auto& s = w.settlements[i];
        if (!s.alive) continue;

        // Food production from adjacent terrain
        float food_gain = 0;
        for (int d = 0; d < 8; d++) {
            int nx = s.x + DX[d], ny = s.y + DY[d];
            if (nx < 0 || nx >= w.W || ny < 0 || ny >= w.H) continue;
            int t = w.grid[ny][nx];
            if (t == TERRAIN_FOREST) food_gain += p.food_per_forest;
            else if (t == TERRAIN_PLAINS || t == TERRAIN_EMPTY) food_gain += p.food_per_plains;
        }
        if (w.is_coastal(s.x, s.y)) food_gain += p.food_per_coastal;
        food_gain *= (1.0f + 0.1f * s.tech);
        s.food += food_gain;
        s.food = std::min(s.food, 1.0f);

        // Population growth
        if (s.food > p.growth_threshold) {
            float growth = p.growth_rate * s.food;
            s.population += growth;
            s.food -= growth * 0.4f;
        }

        // Defense recovery
        if (s.defense < 1.0f) s.defense += 0.02f;
        s.defense = std::min(s.defense, 1.0f);

        // Port development
        if (!s.has_port && s.population > p.port_threshold && w.is_coastal(s.x, s.y)) {
            if (w.randf() < p.port_prob) {
                s.has_port = true;
                w.grid[s.y][s.x] = TERRAIN_PORT;
            }
        }

        // Longship building
        if (s.has_port && s.longships < 2 && s.population > p.longship_threshold) {
            if (w.randf() < p.longship_prob) {
                s.longships++;
            }
        }

        // Expansion: found new settlement
        if (s.population > p.expansion_pop && w.randf() < p.expansion_prob) {
            // Find valid target cells within range, weighted by distance and terrain
            std::vector<std::pair<int,int>> candidates;
            std::vector<float> weights;
            int r = p.expansion_range;
            for (int dy = -r; dy <= r; dy++) {
                for (int dx = -r; dx <= r; dx++) {
                    if (dx == 0 && dy == 0) continue;
                    int tx = s.x + dx, ty = s.y + dy;
                    if (tx < 0 || tx >= w.W || ty < 0 || ty >= w.H) continue;
                    int t = w.grid[ty][tx];
                    // Can expand onto plains, empty, or forest (clearing it)
                    if (t == TERRAIN_PLAINS || t == TERRAIN_EMPTY || t == TERRAIN_FOREST) {
                        if (w.settle_map[ty][tx] == -1) { // not already occupied
                            candidates.push_back({tx, ty});
                            // Weight: prefer closer cells, plains/empty over forest
                            float dist = sqrtf((float)(dx*dx + dy*dy));
                            float terrain_w = (t == TERRAIN_FOREST) ? 0.5f : 1.0f;
                            weights.push_back(terrain_w / std::max(dist, 1.0f));
                        }
                    }
                }
            }
            if (!candidates.empty()) {
                // Weighted random selection
                float total_w = 0;
                for (float wt : weights) total_w += wt;
                float pick = w.randf() * total_w;
                int chosen = 0;
                for (int ci = 0; ci < (int)weights.size(); ci++) {
                    pick -= weights[ci];
                    if (pick <= 0) { chosen = ci; break; }
                }
                auto& [tx, ty] = candidates[chosen];
                expansions.push_back({i, tx, ty});
            }
        }
    }

    // Apply expansions (skip duplicates — two settlements may have picked the same cell)
    for (auto& e : expansions) {
        auto& parent = w.settlements[e.parent_idx];
        if (!parent.alive || parent.population < p.expansion_pop * 0.5f) continue;
        // Race condition guard: skip if cell was already claimed by earlier expansion
        if (w.settle_map[e.ty][e.tx] != -1) continue;

        SimSettlement ns;
        ns.x = e.tx; ns.y = e.ty;
        ns.population = parent.population * 0.25f;
        parent.population -= ns.population;
        ns.food = 0.3f;
        ns.wealth = 0.0f;
        ns.defense = 0.5f;
        ns.tech = parent.tech * 0.5f;
        ns.has_port = false;
        ns.alive = true;
        ns.owner_id = parent.owner_id;
        ns.longships = 0;

        // Check if coastal → could be port
        if (w.is_coastal(e.tx, e.ty) && w.randf() < 0.15f) {
            ns.has_port = true;
            w.grid[e.ty][e.tx] = TERRAIN_PORT;
        } else {
            w.grid[e.ty][e.tx] = TERRAIN_SETTLEMENT;
        }

        w.settle_map[e.ty][e.tx] = (int)w.settlements.size(); // mark cell taken immediately
        w.settlements.push_back(ns);
    }

    if (!expansions.empty()) w.rebuild_settle_map();
}

// ─── Phase 2: Conflict ──────────────────────────────────────────────────────

void phase_conflict(SimWorld& w, const SimParams& p) {
    // Shuffle order
    std::vector<int> order;
    for (int i = 0; i < (int)w.settlements.size(); i++)
        if (w.settlements[i].alive) order.push_back(i);
    std::shuffle(order.begin(), order.end(), w.rng);

    for (int idx : order) {
        auto& s = w.settlements[idx];
        if (!s.alive) continue;

        // Should we raid?
        float raid_prob = p.raid_prob_base;
        if (s.food < p.desperation_food) raid_prob = p.raid_prob_desperate;
        if (w.randf() > raid_prob) continue;

        // Find targets
        float range = (s.longships > 0) ? p.raid_range_sea : p.raid_range_land;
        std::vector<int> targets;
        for (int j = 0; j < (int)w.settlements.size(); j++) {
            if (j == idx || !w.settlements[j].alive) continue;
            if (w.settlements[j].owner_id == s.owner_id) continue;
            float dist = sqrtf((s.x - w.settlements[j].x) * (s.x - w.settlements[j].x) +
                               (s.y - w.settlements[j].y) * (s.y - w.settlements[j].y));
            if (dist <= range) targets.push_back(j);
        }
        if (targets.empty()) continue;

        // Pick target (prefer weaker)
        int tidx = targets[w.rng() % targets.size()];
        auto& target = w.settlements[tidx];

        // Mark these factions as at war (regardless of raid outcome)
        w.mark_war(s.owner_id, target.owner_id);

        // Resolve combat
        float attack = s.population * (1.0f + 0.2f * s.tech);
        float defend = target.defense * target.population;
        float roll = 0.5f + w.randf(); // 0.5-1.5 variance

        if (attack > defend * roll) {
            // Successful raid
            float loot = std::min(target.food * p.raid_loot_frac, 0.2f);
            s.food += loot;
            target.food -= loot;
            s.wealth += 0.01f;
            target.defense -= p.raid_damage;
            target.population -= 0.05f;

            // Conquest?
            if (target.defense < 0.1f && w.randf() < p.conquest_prob) {
                target.owner_id = s.owner_id;
            }
        }
    }
}

// ─── Phase 3: Trade ─────────────────────────────────────────────────────────

void phase_trade(SimWorld& w, const SimParams& p) {
    for (int i = 0; i < (int)w.settlements.size(); i++) {
        auto& a = w.settlements[i];
        if (!a.alive || !a.has_port) continue;

        for (int j = i + 1; j < (int)w.settlements.size(); j++) {
            auto& b = w.settlements[j];
            if (!b.alive || !b.has_port) continue;

            // Trade requires "not at war" — factions that raided each other this year can't trade
            if (w.at_war(a.owner_id, b.owner_id)) continue;

            float dist = sqrtf((a.x - b.x) * (a.x - b.x) + (a.y - b.y) * (a.y - b.y));
            if (dist > p.trade_range) continue;

            // Trade!
            a.food += p.trade_food;
            b.food += p.trade_food;
            a.wealth += p.trade_wealth;
            b.wealth += p.trade_wealth;

            // Tech diffusion
            float avg_tech = (a.tech + b.tech) * 0.5f;
            a.tech += (avg_tech - a.tech) * p.tech_diffusion;
            b.tech += (avg_tech - b.tech) * p.tech_diffusion;
        }
    }
}

// ─── Phase 4: Winter ────────────────────────────────────────────────────────

void phase_winter(SimWorld& w, const SimParams& p) {
    // Winter severity: normal distribution + rare catastrophe
    float severity = p.winter_base_loss + w.randn(0.0f, p.winter_variance);
    if (w.randf() < p.winter_catastrophe_prob) {
        severity *= p.winter_catastrophe_mult;
    }
    severity = std::max(0.05f, std::min(severity, 0.9f));

    std::vector<int> to_collapse;

    for (int i = 0; i < (int)w.settlements.size(); i++) {
        auto& s = w.settlements[i];
        if (!s.alive) continue;

        float food_loss = severity * (1.0f + 0.05f * s.population);
        s.food -= food_loss;

        // Hunger
        if (s.food < 0) {
            s.population += s.food * 0.3f; // food is negative, so this reduces pop
            if (s.population < 0) s.population = 0;
            s.food = 0;
        }

        // Collapse check: population too low, OR starving + weak defense
        if (s.population < p.collapse_pop ||
            (s.food < p.collapse_food + 0.01f && s.defense < p.collapse_defense)) {
            to_collapse.push_back(i);
        }
    }

    // Process collapses
    for (int idx : to_collapse) {
        auto& s = w.settlements[idx];
        s.alive = false;
        w.grid[s.y][s.x] = TERRAIN_RUIN;
        w.settle_map[s.y][s.x] = -1;

        // Disperse population to nearby friendly settlements
        float pop_to_spread = s.population;
        std::vector<int> nearby;
        for (int j = 0; j < (int)w.settlements.size(); j++) {
            if (j == idx || !w.settlements[j].alive) continue;
            if (w.settlements[j].owner_id != s.owner_id) continue;
            float dist = sqrtf((s.x - w.settlements[j].x) * (s.x - w.settlements[j].x) +
                               (s.y - w.settlements[j].y) * (s.y - w.settlements[j].y));
            if (dist <= 5.0f) nearby.push_back(j);
        }
        if (!nearby.empty()) {
            float share = pop_to_spread / nearby.size();
            for (int j : nearby) w.settlements[j].population += share;
        }
    }
}

// ─── Phase 5: Environment ───────────────────────────────────────────────────

void phase_environment(SimWorld& w, const SimParams& p) {
    std::vector<std::tuple<int,int,int>> changes; // x, y, new_terrain

    for (int y = 0; y < w.H; y++) {
        for (int x = 0; x < w.W; x++) {
            if (w.grid[y][x] != TERRAIN_RUIN) continue;

            // Check for reclamation by nearby alive settlement
            int best_idx = -1;
            float best_dist = 999.0f;
            for (int i = 0; i < (int)w.settlements.size(); i++) {
                auto& s = w.settlements[i];
                if (!s.alive) continue;
                float dist = sqrtf((x - s.x) * (x - s.x) + (y - s.y) * (y - s.y));
                if (dist <= p.ruin_reclaim_range && dist < best_dist) {
                    best_dist = dist;
                    best_idx = i;
                }
            }

            if (best_idx >= 0 && w.randf() < p.ruin_reclaim_prob) {
                auto& patron = w.settlements[best_idx];
                SimSettlement ns;
                ns.x = x; ns.y = y;
                ns.population = patron.population * 0.15f;
                ns.food = 0.2f;
                ns.wealth = 0.0f;
                ns.defense = 0.4f;
                ns.tech = patron.tech * 0.3f;
                ns.alive = true;
                ns.owner_id = patron.owner_id;
                ns.longships = 0;

                if (w.is_coastal(x, y) && w.randf() < 0.3f) {
                    ns.has_port = true;
                    changes.push_back({x, y, TERRAIN_PORT});
                } else {
                    ns.has_port = false;
                    changes.push_back({x, y, TERRAIN_SETTLEMENT});
                }
                w.settlements.push_back(ns);
                continue;
            }

            // Forest overgrowth
            int adj_forests = w.count_adjacent(x, y, TERRAIN_FOREST);
            float forest_chance = p.ruin_forest_prob + adj_forests * p.forest_adj_bonus;
            if (w.randf() < forest_chance) {
                changes.push_back({x, y, TERRAIN_FOREST});
                continue;
            }

            // Decay to plains
            if (w.randf() < p.ruin_plains_prob) {
                changes.push_back({x, y, TERRAIN_PLAINS});
            }
        }
    }

    for (auto& [x, y, t] : changes) {
        w.grid[y][x] = t;
    }

    w.rebuild_settle_map();
}

// ─── Run one full simulation ────────────────────────────────────────────────

void simulate_one(SimWorld& w, const SimParams& p) {
    for (int year = 0; year < 50; year++) {
        w.war_pairs.clear();  // Reset war state each year
        phase_growth(w, p);
        phase_conflict(w, p);   // Marks faction pairs at war
        phase_trade(w, p);      // Uses war state: "not at war" = can trade
        phase_winter(w, p);
        phase_environment(w, p);
    }
}

// ─── Parameter jitter ───────────────────────────────────────────────────────
// Add small random perturbations to params each rollout.
// This captures parameter uncertainty — some rollouts simulate "harsher" worlds
// (more collapse, like R3) and others "gentler" (more growth, like R7).
// Without jitter, all rollouts use identical params → missing tail scenarios.

SimParams jitter_params(const SimParams& base, std::mt19937& rng, float jitter_scale) {
    SimParams p = base;
    auto jitter = [&](float val, float lo, float hi) -> float {
        float range = hi - lo;
        float noise = std::normal_distribution<float>(0.0f, jitter_scale * range)(rng);
        return std::max(lo, std::min(hi, val + noise));
    };

    // Growth / expansion params (most impactful for settlement prediction)
    p.expansion_pop      = jitter(p.expansion_pop, 0.5f, 4.0f);
    p.expansion_prob     = jitter(p.expansion_prob, 0.05f, 0.8f);
    p.growth_rate        = jitter(p.growth_rate, 0.02f, 0.25f);
    p.growth_threshold   = jitter(p.growth_threshold, 0.1f, 0.8f);
    p.food_per_forest    = jitter(p.food_per_forest, 0.02f, 0.4f);
    p.food_per_plains    = jitter(p.food_per_plains, 0.01f, 0.15f);

    // Conflict params (raiding drives collapse)
    p.raid_prob_base     = jitter(p.raid_prob_base, 0.01f, 0.25f);
    p.raid_damage        = jitter(p.raid_damage, 0.05f, 0.6f);
    p.conquest_prob      = jitter(p.conquest_prob, 0.02f, 0.35f);

    // Winter params (winter kills settlements)
    p.winter_base_loss   = jitter(p.winter_base_loss, 0.05f, 0.6f);
    p.winter_variance    = jitter(p.winter_variance, 0.02f, 0.3f);
    p.collapse_pop       = jitter(p.collapse_pop, 0.01f, 0.3f);

    // Environment (ruin reclamation)
    p.ruin_reclaim_prob  = jitter(p.ruin_reclaim_prob, 0.05f, 0.5f);
    p.ruin_forest_prob   = jitter(p.ruin_forest_prob, 0.03f, 0.4f);

    // Expansion range (integer — jitter continuous then round)
    {
        float er = jitter((float)p.expansion_range, 1.0f, 8.0f);
        p.expansion_range = std::max(1, std::min(8, (int)roundf(er)));
    }

    return p;
}

// ─── Monte Carlo rollout ────────────────────────────────────────────────────

std::vector<std::vector<std::vector<float>>> monte_carlo(
    const std::vector<std::vector<int>>& initial_grid,
    const SimParams& params, int num_rollouts,
    float jitter_scale = 0.0f) {

    int H = initial_grid.size();
    int W = H > 0 ? initial_grid[0].size() : 0;

    // Accumulate counts per class per cell
    std::vector<std::vector<std::vector<int>>> counts(
        H, std::vector<std::vector<int>>(W, std::vector<int>(NUM_CLASSES, 0)));

    std::mt19937 jitter_rng(12345);

    for (int r = 0; r < num_rollouts; r++) {
        // Optionally jitter params per rollout to capture parameter uncertainty
        SimParams p = (jitter_scale > 0.0f) ?
            jitter_params(params, jitter_rng, jitter_scale) : params;

        SimWorld world = init_world(initial_grid, p, 42 + r * 7919);
        simulate_one(world, p);

        // Tally final state
        for (int y = 0; y < H; y++) {
            for (int x = 0; x < W; x++) {
                int cls = terrain_to_class(world.grid[y][x]);
                counts[y][x][cls]++;
            }
        }

        if ((r + 1) % 100 == 0 || r + 1 == num_rollouts)
            printf("\r  Rollout %d/%d", r + 1, num_rollouts);
    }
    printf("\n");

    // Convert to probabilities with terrain-aware floor
    // Old approach: 0.5% floor on ALL 6 classes → fake mountain/ruin on static cells
    // New approach: only floor classes that are reachable from this cell's initial terrain
    std::vector<std::vector<std::vector<float>>> probs(
        H, std::vector<std::vector<float>>(W, std::vector<float>(NUM_CLASSES)));

    for (int y = 0; y < H; y++) {
        for (int x = 0; x < W; x++) {
            int init_terrain = initial_grid[y][x];
            int init_class = terrain_to_class(init_terrain);

            // Determine which classes are reachable from this cell
            // Mountains never change, ocean never changes
            // Forest/Plains/Empty can become settlement/port/ruin and vice versa
            bool reachable[NUM_CLASSES] = {};
            if (init_terrain == TERRAIN_MOUNTAIN) {
                reachable[5] = true;  // mountain stays mountain
            } else if (init_terrain == TERRAIN_OCEAN) {
                reachable[0] = true;  // ocean stays empty
            } else {
                // Land cells: can become any non-mountain class
                reachable[0] = true;  // empty/plains
                reachable[1] = true;  // settlement
                reachable[2] = true;  // port (if coastal)
                reachable[3] = true;  // ruin
                reachable[4] = true;  // forest
                // Mountain only if it started as mountain
                reachable[5] = false;
            }

            float total = 0;
            for (int c = 0; c < NUM_CLASSES; c++) {
                float raw = (float)counts[y][x][c] / num_rollouts;
                if (reachable[c]) {
                    probs[y][x][c] = std::max(raw, 0.003f);  // small floor for reachable
                } else {
                    probs[y][x][c] = 0.0f;  // impossible class gets zero
                }
                total += probs[y][x][c];
            }
            if (total > 0)
                for (int c = 0; c < NUM_CLASSES; c++) probs[y][x][c] /= total;
        }
    }

    return probs;
}

// I/O, scoring, and apply_floor now in io.hpp
// save_params / load_params in sim_params.hpp

// ─── Main ───────────────────────────────────────────────────────────────────

int main(int argc, char** argv) {
    if (argc < 4) {
        printf("Usage: simulate <grids.bin> <round> <seed> [output.bin]\n");
        printf("  [--rollouts N]   number of Monte Carlo rollouts (default: 500)\n");
        printf("  [--params file]  load calibrated parameters\n");
        printf("  [--jitter F]     param jitter scale 0-1 (default: 0, try 0.05-0.15)\n");
        printf("  [--save-params file] save default parameters\n");
        return 1;
    }

    std::string grids_path = argv[1];
    int round = atoi(argv[2]);
    int seed = atoi(argv[3]);
    std::string output_path = (argc > 4 && argv[4][0] != '-') ? argv[4] : "";
    int num_rollouts = 500;
    float jitter_scale = 0.0f;
    SimParams params;

    // Parse optional args
    for (int i = 4; i < argc; i++) {
        if (!strcmp(argv[i], "--rollouts") && i + 1 < argc)
            num_rollouts = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--params") && i + 1 < argc)
            load_params(argv[++i], params);
        else if (!strcmp(argv[i], "--jitter") && i + 1 < argc)
            jitter_scale = atof(argv[++i]);
        else if (!strcmp(argv[i], "--save-params") && i + 1 < argc)
            save_params(argv[++i], params);
    }

    // Load grid
    std::vector<std::vector<int>> grid;
    int W, H;
    if (!load_grid(grids_path, round, seed, grid, W, H)) return 1;
    printf("Grid [r%d.s%d]: %dx%d\n", round, seed, W, H);

    // Count initial settlements
    int init_settles = 0, init_ports = 0;
    for (int y = 0; y < H; y++)
        for (int x = 0; x < W; x++) {
            if (grid[y][x] == TERRAIN_SETTLEMENT) init_settles++;
            if (grid[y][x] == TERRAIN_PORT) init_ports++;
        }
    printf("  Initial: %d settlements, %d ports\n", init_settles, init_ports);

    // Run Monte Carlo
    printf("Running %d rollouts%s...\n", num_rollouts,
           jitter_scale > 0 ? (", jitter=" + std::to_string(jitter_scale)).c_str() : "");
    auto probs = monte_carlo(grid, params, num_rollouts, jitter_scale);

    // Write output
    if (!output_path.empty()) {
        write_prediction(output_path, round, seed, W, H, probs);
        printf("Written to %s\n", output_path.c_str());
    }

    // Validate against ground truth if available
    std::vector<std::vector<std::vector<float>>> gt;
    int gtW, gtH;
    if (load_ground_truth("data/ground_truth.bin", round, seed, gt, gtW, gtH)) {
        printf("\nValidation vs ground truth:\n");
        score_prediction(gt, probs, W, H);
    }

    return 0;
}
