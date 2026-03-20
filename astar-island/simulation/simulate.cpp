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
#include "database.hpp"
#include "features.hpp"
#include "sim_params.hpp"

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
            // Find valid target cells within range
            std::vector<std::pair<int,int>> candidates;
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
                        }
                    }
                }
            }
            if (!candidates.empty()) {
                // Prefer closer cells and plains over forest
                auto& [tx, ty] = candidates[w.rng() % candidates.size()];
                expansions.push_back({i, tx, ty});
            }
        }
    }

    // Apply expansions
    for (auto& e : expansions) {
        auto& parent = w.settlements[e.parent_idx];
        if (!parent.alive || parent.population < p.expansion_pop * 0.5f) continue;

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
            s.food = 0;
        }

        // Collapse check
        if (s.population < p.collapse_pop ||
            (s.food <= p.collapse_food && s.defense < p.collapse_defense)) {
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
        phase_growth(w, p);
        phase_conflict(w, p);
        phase_trade(w, p);
        phase_winter(w, p);
        phase_environment(w, p);
    }
}

// ─── Monte Carlo rollout ────────────────────────────────────────────────────

std::vector<std::vector<std::vector<float>>> monte_carlo(
    const std::vector<std::vector<int>>& initial_grid,
    const SimParams& params, int num_rollouts) {

    int H = initial_grid.size();
    int W = H > 0 ? initial_grid[0].size() : 0;

    // Accumulate counts per class per cell
    std::vector<std::vector<std::vector<int>>> counts(
        H, std::vector<std::vector<int>>(W, std::vector<int>(NUM_CLASSES, 0)));

    for (int r = 0; r < num_rollouts; r++) {
        SimWorld world = init_world(initial_grid, params, 42 + r * 7919);
        simulate_one(world, params);

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

    // Convert to probabilities with floor
    std::vector<std::vector<std::vector<float>>> probs(
        H, std::vector<std::vector<float>>(W, std::vector<float>(NUM_CLASSES)));

    for (int y = 0; y < H; y++) {
        for (int x = 0; x < W; x++) {
            float total = 0;
            for (int c = 0; c < NUM_CLASSES; c++) {
                probs[y][x][c] = std::max((float)counts[y][x][c] / num_rollouts, 0.005f);
                total += probs[y][x][c];
            }
            for (int c = 0; c < NUM_CLASSES; c++) probs[y][x][c] /= total;
        }
    }

    return probs;
}

// ─── I/O (reuse from predict.cpp) ───────────────────────────────────────────

bool load_grid(const std::string& path, int want_round, int want_seed,
               std::vector<std::vector<int>>& grid, int& W, int& H) {
    std::ifstream f(path, std::ios::binary);
    if (!f) { printf("Error: cannot open %s\n", path.c_str()); return false; }

    char magic[4]; f.read(magic, 4);
    uint16_t ver; f.read((char*)&ver, 2);
    uint32_t count; f.read((char*)&count, 4);

    for (uint32_t i = 0; i < count; i++) {
        int32_t round, seed, w, h;
        f.read((char*)&round, 4);
        f.read((char*)&seed, 4);
        f.read((char*)&w, 4);
        f.read((char*)&h, 4);

        if (round == want_round && seed == want_seed) {
            W = w; H = h;
            grid.resize(H);
            for (int y = 0; y < H; y++) {
                grid[y].resize(W);
                for (int x = 0; x < W; x++) f.read((char*)&grid[y][x], 4);
            }
            return true;
        } else {
            f.seekg((int64_t)w * h * 4, std::ios::cur);
        }
    }
    printf("Grid not found: round=%d seed=%d\n", want_round, want_seed);
    return false;
}

bool load_ground_truth(const std::string& path, int want_round, int want_seed,
                       std::vector<std::vector<std::vector<float>>>& gt, int& W, int& H) {
    std::ifstream f(path, std::ios::binary);
    if (!f) return false;

    char magic[4]; f.read(magic, 4);
    uint16_t ver; f.read((char*)&ver, 2);
    uint32_t count; f.read((char*)&count, 4);

    for (uint32_t i = 0; i < count; i++) {
        int32_t round, seed, w, h;
        f.read((char*)&round, 4);
        f.read((char*)&seed, 4);
        f.read((char*)&w, 4);
        f.read((char*)&h, 4);

        if (round == want_round && seed == want_seed) {
            W = w; H = h;
            gt.resize(H);
            for (int y = 0; y < H; y++) {
                gt[y].resize(W);
                for (int x = 0; x < W; x++) {
                    gt[y][x].resize(NUM_CLASSES);
                    for (int c = 0; c < NUM_CLASSES; c++) f.read((char*)&gt[y][x][c], 4);
                }
            }
            return true;
        } else {
            f.seekg((int64_t)w * h * NUM_CLASSES * 4, std::ios::cur);
        }
    }
    return false;
}

void write_prediction(const std::string& path, int round, int seed, int W, int H,
                      const std::vector<std::vector<std::vector<float>>>& pred) {
    std::ofstream f(path, std::ios::binary);
    f.write("ASTP", 4);
    uint16_t ver = 1;
    f.write((char*)&ver, 2);
    f.write((char*)&round, 4);
    f.write((char*)&seed, 4);
    f.write((char*)&W, 4);
    f.write((char*)&H, 4);
    for (int y = 0; y < H; y++)
        for (int x = 0; x < W; x++)
            for (int c = 0; c < NUM_CLASSES; c++)
                f.write((char*)&pred[y][x][c], 4);
    f.close();
}

// ─── Scoring ────────────────────────────────────────────────────────────────

float score_prediction(const std::vector<std::vector<std::vector<float>>>& gt,
                       const std::vector<std::vector<std::vector<float>>>& pred,
                       int W, int H) {
    double total_entropy = 0, total_weighted_kl = 0;
    int dynamic_cells = 0;

    for (int y = 0; y < H; y++) {
        for (int x = 0; x < W; x++) {
            // Compute entropy
            double ent = 0;
            for (int c = 0; c < NUM_CLASSES; c++)
                if (gt[y][x][c] > 0) ent -= gt[y][x][c] * log(gt[y][x][c]);

            if (ent < 1e-6) continue;
            dynamic_cells++;

            // KL divergence
            double kl = 0;
            for (int c = 0; c < NUM_CLASSES; c++)
                if (gt[y][x][c] > 0)
                    kl += gt[y][x][c] * log(gt[y][x][c] / std::max(pred[y][x][c], 1e-10f));

            total_entropy += ent;
            total_weighted_kl += ent * kl;
        }
    }

    double weighted_kl = total_entropy > 0 ? total_weighted_kl / total_entropy : 0;
    double score = std::max(0.0, std::min(100.0, 100.0 * exp(-3.0 * weighted_kl)));

    printf("  Dynamic cells: %d\n", dynamic_cells);
    printf("  Weighted KL:   %.6f\n", weighted_kl);
    printf("  Score:         %.2f / 100\n", score);
    return (float)score;
}

// save_params / load_params now in sim_params.hpp

// ─── Main ───────────────────────────────────────────────────────────────────

int main(int argc, char** argv) {
    if (argc < 4) {
        printf("Usage: simulate <grids.bin> <round> <seed> [output.bin]\n");
        printf("  [--rollouts N]  number of Monte Carlo rollouts (default: 500)\n");
        printf("  [--params file] load calibrated parameters\n");
        printf("  [--save-params file] save default parameters\n");
        return 1;
    }

    std::string grids_path = argv[1];
    int round = atoi(argv[2]);
    int seed = atoi(argv[3]);
    std::string output_path = (argc > 4 && argv[4][0] != '-') ? argv[4] : "";
    int num_rollouts = 500;
    SimParams params;

    // Parse optional args
    for (int i = 4; i < argc; i++) {
        if (!strcmp(argv[i], "--rollouts") && i + 1 < argc)
            num_rollouts = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--params") && i + 1 < argc)
            load_params(argv[++i], params);
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
    printf("Running %d rollouts...\n", num_rollouts);
    auto probs = monte_carlo(grid, params, num_rollouts);

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
