// calibrate.cpp — CMA-ES parameter calibration for the Norse World Simulator
// Optimizes SimParams to maximize average score across training rounds.
// Usage: ./calibrate data/grids.bin data/training.bin [--generations N] [--output params.bin]

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
#include <thread>
#include <mutex>
#include <atomic>
#include <chrono>
#include <map>
#include "database.hpp"
#include "features.hpp"
#include "sim_params.hpp"

// ─── Parameter vector encoding ───────────────────────────────────────────────

struct ParamDef {
    const char* name;
    float lo, hi;            // bounds
    float default_val;
};

// Which params to optimize and their bounds
static const std::vector<ParamDef> PARAM_DEFS = {
    // Initial state
    {"init_population",     0.2f,  3.0f,  1.0f},
    {"init_food",           0.1f,  1.0f,  0.5f},
    {"init_defense",        0.2f,  2.0f,  0.8f},
    {"init_tech",           0.0f,  0.5f,  0.1f},

    // Growth
    {"food_per_forest",     0.02f, 0.5f,  0.15f},
    {"food_per_plains",     0.005f,0.2f,  0.04f},
    {"food_per_coastal",    0.005f,0.2f,  0.03f},
    {"growth_threshold",    0.1f,  0.8f,  0.4f},
    {"growth_rate",         0.01f, 0.3f,  0.08f},
    {"expansion_pop",       0.8f,  4.0f,  1.8f},
    {"expansion_prob",      0.02f, 0.6f,  0.3f},
    {"port_threshold",      0.5f,  3.0f,  1.3f},
    {"port_prob",           0.02f, 0.5f,  0.25f},
    {"longship_threshold",  0.5f,  3.0f,  1.5f},
    {"longship_prob",       0.02f, 0.4f,  0.15f},

    // Conflict
    {"raid_range_land",     1.0f,  6.0f,  3.0f},
    {"raid_range_sea",      3.0f,  15.0f, 8.0f},
    {"raid_prob_base",      0.01f, 0.3f,  0.08f},
    {"raid_prob_desperate",  0.1f, 0.8f,  0.4f},
    {"desperation_food",    0.02f, 0.4f,  0.15f},
    {"raid_damage",         0.05f, 0.7f,  0.25f},
    {"raid_loot_frac",      0.05f, 0.5f,  0.25f},
    {"conquest_prob",       0.01f, 0.5f,  0.12f},

    // Trade
    {"trade_range",         2.0f,  15.0f, 6.0f},
    {"trade_food",          0.005f,0.2f,  0.04f},
    {"trade_wealth",        0.002f,0.1f,  0.015f},
    {"tech_diffusion",      0.01f, 0.3f,  0.08f},

    // Winter
    {"winter_base_loss",    0.05f, 0.8f,  0.25f},
    {"winter_variance",     0.02f, 0.4f,  0.12f},
    {"winter_catastrophe_prob", 0.01f, 0.3f, 0.05f},
    {"winter_catastrophe_mult", 1.0f, 5.0f, 2.5f},
    {"collapse_pop",        0.01f, 0.5f,  0.08f},
    {"collapse_food",       0.0f,  0.3f,  0.0f},
    {"collapse_defense",    0.02f, 0.5f,  0.15f},

    // Environment
    {"ruin_reclaim_range",  0.5f,  5.0f,  2.5f},
    {"ruin_reclaim_prob",   0.02f, 0.5f,  0.25f},
    {"ruin_forest_prob",    0.02f, 0.5f,  0.15f},
    {"ruin_plains_prob",    0.01f, 0.3f,  0.08f},
    {"forest_adj_bonus",    0.005f,0.2f,  0.05f},

    // Previously hardcoded — now calibrated
    {"expansion_range",     1.0f,  8.0f,  3.0f},
};

static const int N_PARAMS = PARAM_DEFS.size();

// Convert flat vector to SimParams
SimParams vec_to_params(const std::vector<double>& v) {
    SimParams p;
    int i = 0;
    p.init_population    = v[i++];
    p.init_food          = v[i++];
    p.init_defense       = v[i++];
    p.init_tech          = v[i++];

    p.food_per_forest    = v[i++];
    p.food_per_plains    = v[i++];
    p.food_per_coastal   = v[i++];
    p.growth_threshold   = v[i++];
    p.growth_rate        = v[i++];
    p.expansion_pop      = v[i++];
    p.expansion_prob     = v[i++];
    p.port_threshold     = v[i++];
    p.port_prob          = v[i++];
    p.longship_threshold = v[i++];
    p.longship_prob      = v[i++];

    p.raid_range_land    = v[i++];
    p.raid_range_sea     = v[i++];
    p.raid_prob_base     = v[i++];
    p.raid_prob_desperate = v[i++];
    p.desperation_food   = v[i++];
    p.raid_damage        = v[i++];
    p.raid_loot_frac     = v[i++];
    p.conquest_prob      = v[i++];

    p.trade_range        = v[i++];
    p.trade_food         = v[i++];
    p.trade_wealth       = v[i++];
    p.tech_diffusion     = v[i++];

    p.winter_base_loss   = v[i++];
    p.winter_variance    = v[i++];
    p.winter_catastrophe_prob = v[i++];
    p.winter_catastrophe_mult = v[i++];
    p.collapse_pop       = v[i++];
    p.collapse_food      = v[i++];
    p.collapse_defense   = v[i++];

    p.ruin_reclaim_range = v[i++];
    p.ruin_reclaim_prob  = v[i++];
    p.ruin_forest_prob   = v[i++];
    p.ruin_plains_prob   = v[i++];
    p.forest_adj_bonus   = v[i++];

    p.expansion_range = std::max(1, (int)roundf(v[i++])); // continuous → int
    return p;
}

std::vector<double> params_to_vec(const SimParams& p) {
    return {
        p.init_population, p.init_food, p.init_defense, p.init_tech,
        p.food_per_forest, p.food_per_plains, p.food_per_coastal,
        p.growth_threshold, p.growth_rate, p.expansion_pop, p.expansion_prob,
        p.port_threshold, p.port_prob, p.longship_threshold, p.longship_prob,
        p.raid_range_land, p.raid_range_sea, p.raid_prob_base, p.raid_prob_desperate,
        p.desperation_food, p.raid_damage, p.raid_loot_frac, p.conquest_prob,
        p.trade_range, p.trade_food, p.trade_wealth, p.tech_diffusion,
        p.winter_base_loss, p.winter_variance, p.winter_catastrophe_prob,
        p.winter_catastrophe_mult, p.collapse_pop, p.collapse_food, p.collapse_defense,
        p.ruin_reclaim_range, p.ruin_reclaim_prob, p.ruin_forest_prob, p.ruin_plains_prob,
        p.forest_adj_bonus,
        (double)p.expansion_range  // now calibrated
    };
}

// Clip to bounds
void clip_to_bounds(std::vector<double>& v) {
    for (int i = 0; i < N_PARAMS; i++) {
        v[i] = std::max((double)PARAM_DEFS[i].lo, std::min((double)PARAM_DEFS[i].hi, v[i]));
    }
}

// ─── Simulation code (duplicated for self-containment) ──────────────────────

struct SimSettlement {
    int x, y;
    float population, food, wealth, defense, tech;
    bool has_port, alive;
    int owner_id, longships;
};

struct SimWorld {
    int W, H;
    std::vector<std::vector<int>> grid;
    std::vector<SimSettlement> settlements;
    std::mt19937 rng;
    std::vector<std::vector<int>> settle_map;
    std::set<std::pair<int,int>> war_pairs;

    void mark_war(int fa, int fb) {
        if (fa == fb) return;
        war_pairs.insert({std::min(fa,fb), std::max(fa,fb)});
    }
    bool at_war(int fa, int fb) const {
        if (fa == fb) return false;
        return war_pairs.count({std::min(fa,fb), std::max(fa,fb)}) > 0;
    }

    void rebuild_settle_map() {
        settle_map.assign(H, std::vector<int>(W, -1));
        for (int i = 0; i < (int)settlements.size(); i++)
            if (settlements[i].alive) settle_map[settlements[i].y][settlements[i].x] = i;
    }
    bool is_land(int x, int y) const {
        if (x < 0 || x >= W || y < 0 || y >= H) return false;
        return grid[y][x] != TERRAIN_OCEAN;
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

SimWorld init_world(const std::vector<std::vector<int>>& initial_grid,
                    const SimParams& p, uint32_t seed) {
    SimWorld w;
    w.H = initial_grid.size();
    w.W = w.H > 0 ? initial_grid[0].size() : 0;
    w.rng.seed(seed);
    w.grid = initial_grid;
    w.settlements.reserve(w.W * w.H); // prevent reallocation invalidating references
    int next_owner = 0;
    for (int y = 0; y < w.H; y++) {
        for (int x = 0; x < w.W; x++) {
            int t = w.grid[y][x];
            if (t == TERRAIN_SETTLEMENT || t == TERRAIN_PORT) {
                SimSettlement s;
                s.x = x; s.y = y;
                s.population = p.init_population * (0.8f + w.randf() * 0.4f);
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

void phase_growth(SimWorld& w, const SimParams& p) {
    struct EC { int pi, tx, ty; };
    std::vector<EC> expansions;

    for (int i = 0; i < (int)w.settlements.size(); i++) {
        auto& s = w.settlements[i];
        if (!s.alive) continue;
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
        if (s.food > p.growth_threshold) {
            float growth = p.growth_rate * s.food;
            s.population += growth;
            s.food -= growth * 0.4f;
        }
        if (s.defense < 1.0f) s.defense += 0.02f;
        s.defense = std::min(s.defense, 1.0f);
        if (!s.has_port && s.population > p.port_threshold && w.is_coastal(s.x, s.y)) {
            if (w.randf() < p.port_prob) { s.has_port = true; w.grid[s.y][s.x] = TERRAIN_PORT; }
        }
        if (s.has_port && s.longships < 2 && s.population > p.longship_threshold) {
            if (w.randf() < p.longship_prob) s.longships++;
        }
        if (s.population > p.expansion_pop && w.randf() < p.expansion_prob) {
            std::vector<std::pair<int,int>> cands;
            int r = p.expansion_range;
            for (int dy = -r; dy <= r; dy++) for (int dx = -r; dx <= r; dx++) {
                if (dx == 0 && dy == 0) continue;
                int tx = s.x + dx, ty = s.y + dy;
                if (tx < 0 || tx >= w.W || ty < 0 || ty >= w.H) continue;
                int t = w.grid[ty][tx];
                if ((t == TERRAIN_PLAINS || t == TERRAIN_EMPTY || t == TERRAIN_FOREST) && w.settle_map[ty][tx] == -1)
                    cands.push_back({tx, ty});
            }
            if (!cands.empty()) {
                auto& [tx, ty] = cands[w.rng() % cands.size()];
                expansions.push_back({i, tx, ty});
            }
        }
    }
    for (auto& e : expansions) {
        auto& parent = w.settlements[e.pi];
        if (!parent.alive || parent.population < p.expansion_pop * 0.5f) continue;
        SimSettlement ns;
        ns.x = e.tx; ns.y = e.ty;
        ns.population = parent.population * 0.25f;
        parent.population -= ns.population;
        ns.food = 0.3f; ns.wealth = 0.0f; ns.defense = 0.5f;
        ns.tech = parent.tech * 0.5f;
        ns.has_port = false; ns.alive = true; ns.owner_id = parent.owner_id; ns.longships = 0;
        if (w.is_coastal(e.tx, e.ty) && w.randf() < 0.15f) {
            ns.has_port = true; w.grid[e.ty][e.tx] = TERRAIN_PORT;
        } else { w.grid[e.ty][e.tx] = TERRAIN_SETTLEMENT; }
        w.settlements.push_back(ns);
    }
    if (!expansions.empty()) w.rebuild_settle_map();
}

void phase_conflict(SimWorld& w, const SimParams& p) {
    std::vector<int> order;
    for (int i = 0; i < (int)w.settlements.size(); i++)
        if (w.settlements[i].alive) order.push_back(i);
    std::shuffle(order.begin(), order.end(), w.rng);
    for (int idx : order) {
        auto& s = w.settlements[idx];
        if (!s.alive) continue;
        float raid_prob = p.raid_prob_base;
        if (s.food < p.desperation_food) raid_prob = p.raid_prob_desperate;
        if (w.randf() > raid_prob) continue;
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
        int tidx = targets[w.rng() % targets.size()];
        auto& target = w.settlements[tidx];
        w.mark_war(s.owner_id, target.owner_id);
        float attack = s.population * (1.0f + 0.2f * s.tech);
        float defend = target.defense * target.population;
        float roll = 0.5f + w.randf();
        if (attack > defend * roll) {
            float loot = std::min(target.food * p.raid_loot_frac, 0.2f);
            s.food += loot; target.food -= loot;
            s.wealth += 0.01f; target.defense -= p.raid_damage; target.population -= 0.05f;
            if (target.defense < 0.1f && w.randf() < p.conquest_prob)
                target.owner_id = s.owner_id;
        }
    }
}

void phase_trade(SimWorld& w, const SimParams& p) {
    for (int i = 0; i < (int)w.settlements.size(); i++) {
        auto& a = w.settlements[i];
        if (!a.alive || !a.has_port) continue;
        for (int j = i + 1; j < (int)w.settlements.size(); j++) {
            auto& b = w.settlements[j];
            if (!b.alive || !b.has_port) continue;
            if (w.at_war(a.owner_id, b.owner_id)) continue;
            float dist = sqrtf((a.x - b.x) * (a.x - b.x) + (a.y - b.y) * (a.y - b.y));
            if (dist > p.trade_range) continue;
            a.food += p.trade_food; b.food += p.trade_food;
            a.wealth += p.trade_wealth; b.wealth += p.trade_wealth;
            float avg_tech = (a.tech + b.tech) * 0.5f;
            a.tech += (avg_tech - a.tech) * p.tech_diffusion;
            b.tech += (avg_tech - b.tech) * p.tech_diffusion;
        }
    }
}

void phase_winter(SimWorld& w, const SimParams& p) {
    float severity = p.winter_base_loss + w.randn(0.0f, p.winter_variance);
    if (w.randf() < p.winter_catastrophe_prob) severity *= p.winter_catastrophe_mult;
    severity = std::max(0.05f, std::min(severity, 0.9f));
    std::vector<int> to_collapse;
    for (int i = 0; i < (int)w.settlements.size(); i++) {
        auto& s = w.settlements[i];
        if (!s.alive) continue;
        float food_loss = severity * (1.0f + 0.05f * s.population);
        s.food -= food_loss;
        if (s.food < 0) {
            s.population += s.food * 0.3f;
            s.food = 0;
        }
        if (s.population < p.collapse_pop || (s.food <= p.collapse_food && s.defense < p.collapse_defense))
            to_collapse.push_back(i);
    }
    for (int idx : to_collapse) {
        auto& s = w.settlements[idx];
        s.alive = false;
        w.grid[s.y][s.x] = TERRAIN_RUIN;
        w.settle_map[s.y][s.x] = -1;
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

void phase_environment(SimWorld& w, const SimParams& p) {
    std::vector<std::tuple<int,int,int>> changes;
    for (int y = 0; y < w.H; y++) for (int x = 0; x < w.W; x++) {
        if (w.grid[y][x] != TERRAIN_RUIN) continue;
        int best_idx = -1; float best_dist = 999.0f;
        for (int i = 0; i < (int)w.settlements.size(); i++) {
            auto& s = w.settlements[i];
            if (!s.alive) continue;
            float dist = sqrtf((x - s.x) * (x - s.x) + (y - s.y) * (y - s.y));
            if (dist <= p.ruin_reclaim_range && dist < best_dist) { best_dist = dist; best_idx = i; }
        }
        if (best_idx >= 0 && w.randf() < p.ruin_reclaim_prob) {
            auto& patron = w.settlements[best_idx];
            SimSettlement ns;
            ns.x = x; ns.y = y;
            ns.population = patron.population * 0.15f;
            ns.food = 0.2f; ns.wealth = 0.0f; ns.defense = 0.4f;
            ns.tech = patron.tech * 0.3f; ns.alive = true; ns.owner_id = patron.owner_id; ns.longships = 0;
            if (w.is_coastal(x, y) && w.randf() < 0.3f) {
                ns.has_port = true; changes.push_back({x, y, TERRAIN_PORT});
            } else {
                ns.has_port = false; changes.push_back({x, y, TERRAIN_SETTLEMENT});
            }
            w.settlements.push_back(ns);
            continue;
        }
        int adj_forests = w.count_adjacent(x, y, TERRAIN_FOREST);
        float forest_chance = p.ruin_forest_prob + adj_forests * p.forest_adj_bonus;
        if (w.randf() < forest_chance) { changes.push_back({x, y, TERRAIN_FOREST}); continue; }
        if (w.randf() < p.ruin_plains_prob) changes.push_back({x, y, TERRAIN_PLAINS});
    }
    for (auto& [x, y, t] : changes) w.grid[y][x] = t;
    w.rebuild_settle_map();
}

void simulate_one(SimWorld& w, const SimParams& p) {
    for (int year = 0; year < 50; year++) {
        w.war_pairs.clear();
        phase_growth(w, p);
        phase_conflict(w, p);
        phase_trade(w, p);
        phase_winter(w, p);
        phase_environment(w, p);
        // Safety: cap settlements to prevent runaway growth
        if ((int)w.settlements.size() > w.W * w.H * 2) break;
    }
}

// ─── Monte Carlo (returns weighted KL, lower is better) ─────────────────────

// Lightweight jitter for calibration — matches simulate.cpp's jitter_params
SimParams jitter_params_cal(const SimParams& base, std::mt19937& rng, float scale = 0.08f) {
    SimParams p = base;
    auto jit = [&](float val, float lo, float hi) -> float {
        float noise = std::normal_distribution<float>(0.0f, scale * (hi - lo))(rng);
        return std::max(lo, std::min(hi, val + noise));
    };
    p.expansion_pop      = jit(p.expansion_pop, 0.5f, 4.0f);
    p.expansion_prob     = jit(p.expansion_prob, 0.05f, 0.8f);
    p.growth_rate        = jit(p.growth_rate, 0.02f, 0.25f);
    p.growth_threshold   = jit(p.growth_threshold, 0.1f, 0.8f);
    p.food_per_forest    = jit(p.food_per_forest, 0.02f, 0.4f);
    p.food_per_plains    = jit(p.food_per_plains, 0.01f, 0.15f);
    p.raid_prob_base     = jit(p.raid_prob_base, 0.01f, 0.25f);
    p.raid_damage        = jit(p.raid_damage, 0.05f, 0.6f);
    p.conquest_prob      = jit(p.conquest_prob, 0.02f, 0.35f);
    p.winter_base_loss   = jit(p.winter_base_loss, 0.05f, 0.6f);
    p.winter_variance    = jit(p.winter_variance, 0.02f, 0.3f);
    p.collapse_pop       = jit(p.collapse_pop, 0.01f, 0.3f);
    p.ruin_reclaim_prob  = jit(p.ruin_reclaim_prob, 0.05f, 0.5f);
    p.ruin_forest_prob   = jit(p.ruin_forest_prob, 0.03f, 0.4f);
    { float er = jit((float)p.expansion_range, 1.0f, 8.0f);
      p.expansion_range = std::max(1, std::min(8, (int)roundf(er))); }
    return p;
}

float evaluate_params(const std::vector<std::vector<int>>& grid,
                      const std::vector<std::vector<std::vector<float>>>& gt,
                      const SimParams& params, int num_rollouts) {
    int H = grid.size(), W = grid[0].size();

    // Accumulate counts with jitter (matches how simulate.cpp actually runs)
    std::vector<std::vector<std::vector<int>>> counts(
        H, std::vector<std::vector<int>>(W, std::vector<int>(NUM_CLASSES, 0)));

    std::mt19937 jitter_rng(12345);
    for (int r = 0; r < num_rollouts; r++) {
        SimParams jp = jitter_params_cal(params, jitter_rng);
        SimWorld world = init_world(grid, jp, 42 + r * 7919);
        simulate_one(world, jp);
        for (int y = 0; y < H; y++)
            for (int x = 0; x < W; x++)
                counts[y][x][terrain_to_class(world.grid[y][x])]++;
    }

    // Convert to probabilities with terrain-aware floor (matches simulate.cpp)
    std::vector<std::vector<std::vector<float>>> probs(
        H, std::vector<std::vector<float>>(W, std::vector<float>(NUM_CLASSES)));
    for (int y = 0; y < H; y++) for (int x = 0; x < W; x++) {
        int init_terrain = grid[y][x];
        float total = 0;
        for (int c = 0; c < NUM_CLASSES; c++) {
            float raw = (float)counts[y][x][c] / num_rollouts;
            bool reachable = (init_terrain == TERRAIN_MOUNTAIN) ? (c == 5)
                           : (init_terrain == TERRAIN_OCEAN) ? (c == 0)
                           : (c != 5); // land: all except mountain
            probs[y][x][c] = reachable ? std::max(raw, 0.003f) : 0.0f;
            total += probs[y][x][c];
        }
        if (total > 0) for (int c = 0; c < NUM_CLASSES; c++) probs[y][x][c] /= total;
    }

    // Score
    double total_entropy = 0, total_weighted_kl = 0;
    for (int y = 0; y < H; y++) for (int x = 0; x < W; x++) {
        double ent = 0;
        for (int c = 0; c < NUM_CLASSES; c++)
            if (gt[y][x][c] > 0) ent -= gt[y][x][c] * log(gt[y][x][c]);
        if (ent < 1e-6) continue;
        double kl = 0;
        for (int c = 0; c < NUM_CLASSES; c++)
            if (gt[y][x][c] > 0)
                kl += gt[y][x][c] * log(gt[y][x][c] / std::max(probs[y][x][c], 1e-10f));
        total_entropy += ent;
        total_weighted_kl += ent * kl;
    }
    return total_entropy > 0 ? total_weighted_kl / total_entropy : 10.0;
}

// ─── Data loading ───────────────────────────────────────────────────────────

struct TrainingCase {
    int round, seed;
    std::vector<std::vector<int>> grid;
    std::vector<std::vector<std::vector<float>>> gt;
    int W, H;
};

bool load_all_data(const std::string& grids_path, const std::string& gt_path,
                   std::vector<TrainingCase>& cases) {
    // Load grids
    std::ifstream gf(grids_path, std::ios::binary);
    if (!gf) { printf("Cannot open %s\n", grids_path.c_str()); return false; }
    char magic[4]; gf.read(magic, 4);
    uint16_t ver; gf.read((char*)&ver, 2);
    uint32_t count; gf.read((char*)&count, 4);

    struct GridEntry { int round, seed, W, H; std::vector<std::vector<int>> grid; };
    std::vector<GridEntry> grids;
    for (uint32_t i = 0; i < count; i++) {
        GridEntry e;
        gf.read((char*)&e.round, 4); gf.read((char*)&e.seed, 4);
        gf.read((char*)&e.W, 4); gf.read((char*)&e.H, 4);
        e.grid.resize(e.H);
        for (int y = 0; y < e.H; y++) {
            e.grid[y].resize(e.W);
            for (int x = 0; x < e.W; x++) gf.read((char*)&e.grid[y][x], 4);
        }
        grids.push_back(std::move(e));
    }

    // Load ground truth
    std::ifstream tf(gt_path, std::ios::binary);
    if (!tf) { printf("Cannot open %s\n", gt_path.c_str()); return false; }
    tf.read(magic, 4); tf.read((char*)&ver, 2); tf.read((char*)&count, 4);

    struct GTEntry { int round, seed, W, H; std::vector<std::vector<std::vector<float>>> gt; };
    std::vector<GTEntry> gts;
    for (uint32_t i = 0; i < count; i++) {
        GTEntry e;
        tf.read((char*)&e.round, 4); tf.read((char*)&e.seed, 4);
        tf.read((char*)&e.W, 4); tf.read((char*)&e.H, 4);
        e.gt.resize(e.H);
        for (int y = 0; y < e.H; y++) {
            e.gt[y].resize(e.W);
            for (int x = 0; x < e.W; x++) {
                e.gt[y][x].resize(NUM_CLASSES);
                for (int c = 0; c < NUM_CLASSES; c++) tf.read((char*)&e.gt[y][x][c], 4);
            }
        }
        gts.push_back(std::move(e));
    }

    // Match grids to ground truth (only take first match per GT to avoid moved-from data)
    std::vector<bool> gt_used(gts.size(), false);
    for (auto& g : grids) {
        for (size_t ti = 0; ti < gts.size(); ti++) {
            if (gt_used[ti]) continue;
            auto& t = gts[ti];
            if (g.round == t.round && g.seed == t.seed) {
                TrainingCase tc;
                tc.round = g.round; tc.seed = g.seed;
                tc.W = g.W; tc.H = g.H;
                tc.grid = std::move(g.grid);
                tc.gt = std::move(t.gt);
                cases.push_back(std::move(tc));
                gt_used[ti] = true;
                break;
            }
        }
    }

    printf("Loaded %d training cases\n", (int)cases.size());
    return !cases.empty();
}

// ─── CMA-ES Implementation ─────────────────────────────────────────────────

// Simplified CMA-ES (sep-CMA-ES using diagonal covariance for efficiency)
// Good enough for ~40 parameters and much faster than full CMA-ES

struct CMAES {
    int n;                          // dimension
    int lambda;                     // population size
    int mu;                         // number of parents
    double sigma;                   // step size
    std::vector<double> mean;       // current mean
    std::vector<double> diagC;      // diagonal of covariance matrix
    std::vector<double> ps;         // evolution path for sigma
    std::vector<double> pc;         // evolution path for C
    std::vector<double> weights;    // recombination weights
    double mueff;                   // variance-effective size of mu
    double cs, ds, cc, c1, cmu;    // strategy parameters

    std::mt19937 rng;

    CMAES(int dim, const std::vector<double>& initial_mean, double initial_sigma, int seed = 42)
        : n(dim), sigma(initial_sigma), mean(initial_mean), rng(seed) {

        lambda = 4 + (int)(3 * log(n));  // population size
        mu = lambda / 2;

        // Weights
        weights.resize(mu);
        double sum_w = 0;
        for (int i = 0; i < mu; i++) {
            weights[i] = log(mu + 0.5) - log(i + 1);
            sum_w += weights[i];
        }
        for (int i = 0; i < mu; i++) weights[i] /= sum_w;

        // mueff
        double sum_w2 = 0;
        for (int i = 0; i < mu; i++) sum_w2 += weights[i] * weights[i];
        mueff = 1.0 / sum_w2;

        // Strategy parameters
        cs = (mueff + 2) / (n + mueff + 5);
        ds = 1 + 2 * std::max(0.0, sqrt((mueff - 1) / (n + 1)) - 1) + cs;
        cc = (4 + mueff / n) / (n + 4 + 2 * mueff / n);
        c1 = 2 / ((n + 1.3) * (n + 1.3) + mueff);
        cmu = std::min(1 - c1, 2 * (mueff - 2 + 1.0 / mueff) / ((n + 2) * (n + 2) + mueff));

        // Initialize
        diagC.assign(n, 1.0);
        ps.assign(n, 0.0);
        pc.assign(n, 0.0);
    }

    // Sample lambda candidates
    std::vector<std::vector<double>> sample() {
        std::vector<std::vector<double>> pop(lambda, std::vector<double>(n));
        std::normal_distribution<double> norm(0, 1);
        for (int i = 0; i < lambda; i++) {
            for (int j = 0; j < n; j++) {
                pop[i][j] = mean[j] + sigma * sqrt(diagC[j]) * norm(rng);
            }
        }
        return pop;
    }

    // Update with sorted population (best first)
    void update(const std::vector<std::vector<double>>& sorted_pop) {
        // New mean
        std::vector<double> old_mean = mean;
        for (int j = 0; j < n; j++) {
            mean[j] = 0;
            for (int i = 0; i < mu; i++)
                mean[j] += weights[i] * sorted_pop[i][j];
        }

        // Evolution path for sigma (ps)
        double chi_n = sqrt((double)n) * (1 - 1.0 / (4 * n) + 1.0 / (21 * n * n));
        for (int j = 0; j < n; j++) {
            ps[j] = (1 - cs) * ps[j] + sqrt(cs * (2 - cs) * mueff) * (mean[j] - old_mean[j]) / (sigma * sqrt(diagC[j]));
        }

        // sigma adaptation
        double ps_norm = 0;
        for (int j = 0; j < n; j++) ps_norm += ps[j] * ps[j];
        ps_norm = sqrt(ps_norm);
        sigma *= exp(cs / ds * (ps_norm / chi_n - 1));

        // Hsig
        double ps_norm2 = ps_norm * ps_norm;
        int gen_approx = 1; // simplified
        double threshold = (1.4 + 2.0 / (n + 1)) * chi_n;
        int hsig = ps_norm / sqrt(1 - pow(1 - cs, 2 * gen_approx)) < threshold ? 1 : 0;

        // Evolution path for C (pc)
        for (int j = 0; j < n; j++) {
            pc[j] = (1 - cc) * pc[j] + hsig * sqrt(cc * (2 - cc) * mueff) * (mean[j] - old_mean[j]) / sigma;
        }

        // Covariance matrix update (diagonal)
        for (int j = 0; j < n; j++) {
            double rank_mu_update = 0;
            for (int i = 0; i < mu; i++) {
                double dj = (sorted_pop[i][j] - old_mean[j]) / sigma;
                rank_mu_update += weights[i] * dj * dj;
            }
            diagC[j] = (1 - c1 - cmu) * diagC[j] + c1 * (pc[j] * pc[j] + (1 - hsig) * cc * (2 - cc) * diagC[j]) + cmu * rank_mu_update;
            diagC[j] = std::max(1e-20, diagC[j]);
        }

        // Bound sigma
        sigma = std::max(1e-10, std::min(sigma, 10.0));
    }
};

// save_params / load_params now in sim_params.hpp

int main(int argc, char** argv) {
    if (argc < 3) {
        printf("Usage: calibrate <grids.bin> <ground_truth.bin> [options]\n");
        printf("  --generations N     CMA-ES generations (default: 150)\n");
        printf("  --rollouts N        rollouts per evaluation (default: 30)\n");
        printf("  --output file       save best params (default: data/params.bin)\n");
        printf("  --threads N         parallel threads (default: auto)\n");
        printf("  --sigma N           initial sigma (default: 0.3)\n");
        printf("  --cma-seed N        random seed for CMA-ES (default: 42)\n");
        printf("  --init-params file  warm-start from previous params\n");
        printf("  --seeds-per-round N training seeds per round (default: 3)\n");
        return 1;
    }

    std::string grids_path = argv[1];
    std::string gt_path = argv[2];
    int generations = 150;
    int rollouts = 30;
    std::string output = "data/params.bin";
    int n_threads = std::max(1u, std::thread::hardware_concurrency());
    double initial_sigma = 0.3;
    int cma_seed = 42;
    std::string init_params_path;
    int seeds_per_round = 3;

    for (int i = 3; i < argc; i++) {
        if (!strcmp(argv[i], "--generations") && i+1 < argc) generations = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--rollouts") && i+1 < argc) rollouts = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--output") && i+1 < argc) output = argv[++i];
        else if (!strcmp(argv[i], "--threads") && i+1 < argc) n_threads = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--sigma") && i+1 < argc) initial_sigma = atof(argv[++i]);
        else if (!strcmp(argv[i], "--cma-seed") && i+1 < argc) cma_seed = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--init-params") && i+1 < argc) init_params_path = argv[++i];
        else if (!strcmp(argv[i], "--seeds-per-round") && i+1 < argc) seeds_per_round = atoi(argv[++i]);
    }

    printf("CMA-ES Calibration\n");
    printf("  Generations:     %d\n", generations);
    printf("  Rollouts:        %d (per evaluation)\n", rollouts);
    printf("  Threads:         %d\n", n_threads);
    printf("  Sigma:           %.3f\n", initial_sigma);
    printf("  CMA-ES seed:     %d\n", cma_seed);
    printf("  Params:          %d\n", N_PARAMS);
    printf("  Seeds/round:     %d\n", seeds_per_round);
    printf("  Output:          %s\n", output.c_str());
    if (!init_params_path.empty())
        printf("  Init params:     %s\n", init_params_path.c_str());

    // Load training data
    std::vector<TrainingCase> all_cases;
    if (!load_all_data(grids_path, gt_path, all_cases)) return 1;

    // Use a subset for speed
    std::vector<TrainingCase*> train_cases;
    std::map<int, int> per_round;
    for (auto& tc : all_cases) {
        if (per_round[tc.round] < seeds_per_round) {
            train_cases.push_back(&tc);
            per_round[tc.round]++;
        }
    }
    printf("Using %d training cases for calibration\n", (int)train_cases.size());

    // Initialize starting point
    SimParams start_params;
    if (!init_params_path.empty()) {
        if (!load_params(init_params_path, start_params)) {
            printf("Warning: could not load %s, using defaults\n", init_params_path.c_str());
        }
    }
    std::vector<double> init_vec = params_to_vec(start_params);

    // Normalize to [0,1] for CMA-ES
    std::vector<double> normalized(N_PARAMS);
    for (int i = 0; i < N_PARAMS; i++) {
        normalized[i] = (init_vec[i] - PARAM_DEFS[i].lo) / (PARAM_DEFS[i].hi - PARAM_DEFS[i].lo);
    }

    CMAES cma(N_PARAMS, normalized, initial_sigma, cma_seed);
    printf("Population size: %d, mu: %d\n", cma.lambda, cma.mu);

    // Track best
    double best_score_ever = 0;
    std::vector<double> best_vec_ever = init_vec;

    auto start_time = std::chrono::steady_clock::now();

    for (int gen = 0; gen < generations; gen++) {
        auto gen_start = std::chrono::steady_clock::now();

        // Sample population (in normalized [0,1] space)
        auto pop_normalized = cma.sample();

        // Denormalize and clip
        std::vector<std::vector<double>> pop_real(cma.lambda, std::vector<double>(N_PARAMS));
        for (int i = 0; i < cma.lambda; i++) {
            for (int j = 0; j < N_PARAMS; j++) {
                double nv = std::max(0.0, std::min(1.0, pop_normalized[i][j]));
                pop_real[i][j] = PARAM_DEFS[j].lo + nv * (PARAM_DEFS[j].hi - PARAM_DEFS[j].lo);
                pop_normalized[i][j] = nv;  // clip normalized too
            }
        }

        // Evaluate fitness (parallel)
        std::vector<double> fitness(cma.lambda);
        std::mutex print_mtx;

        std::atomic<int> completed{0};
        auto eval_one = [&](int idx) {
            SimParams p = vec_to_params(pop_real[idx]);
            double total_wkl = 0;
            int n_cases = 0;
            for (auto* tc : train_cases) {
                double wkl = evaluate_params(tc->grid, tc->gt, p, rollouts);
                total_wkl += wkl;
                n_cases++;
            }
            // Fitness = average score (higher is better)
            double avg_wkl = total_wkl / n_cases;
            double score = std::max(0.0, std::min(100.0, 100.0 * exp(-3.0 * avg_wkl)));
            fitness[idx] = score;
            int done = ++completed;
            if (done % 4 == 0 || done == cma.lambda) {
                std::lock_guard<std::mutex> lock(print_mtx);
                printf("  eval %d/%d\r", done, cma.lambda);
                fflush(stdout);
            }
        };

        // Thread pool
        std::vector<std::thread> threads;
        int next_task = 0;
        std::mutex task_mtx;

        for (int t = 0; t < n_threads; t++) {
            threads.emplace_back([&]() {
                while (true) {
                    int task;
                    {
                        std::lock_guard<std::mutex> lock(task_mtx);
                        if (next_task >= cma.lambda) break;
                        task = next_task++;
                    }
                    eval_one(task);
                }
            });
        }
        for (auto& t : threads) t.join();

        // Sort by fitness (descending = best first)
        std::vector<int> order(cma.lambda);
        std::iota(order.begin(), order.end(), 0);
        std::sort(order.begin(), order.end(), [&](int a, int b) {
            return fitness[a] > fitness[b];
        });

        // Reorder normalized population
        std::vector<std::vector<double>> sorted_pop(cma.lambda);
        for (int i = 0; i < cma.lambda; i++) sorted_pop[i] = pop_normalized[order[i]];

        // CMA-ES minimizes, but we sorted best-first, so just pass to update
        // Actually CMA-ES update expects best-first (lowest cost = highest fitness)
        cma.update(sorted_pop);

        double best_fitness = fitness[order[0]];
        double worst_fitness = fitness[order[cma.lambda - 1]];
        double mean_fitness = 0;
        for (int i = 0; i < cma.lambda; i++) mean_fitness += fitness[i];
        mean_fitness /= cma.lambda;

        // Track global best
        if (best_fitness > best_score_ever) {
            best_score_ever = best_fitness;
            best_vec_ever = pop_real[order[0]];
        }

        auto gen_end = std::chrono::steady_clock::now();
        double gen_secs = std::chrono::duration<double>(gen_end - gen_start).count();
        double total_secs = std::chrono::duration<double>(gen_end - start_time).count();

        printf("Gen %3d/%d  best=%.2f  mean=%.2f  worst=%.2f  sigma=%.4f  (%.1fs, total %.0fs) BEST_EVER=%.2f\n",
               gen + 1, generations, best_fitness, mean_fitness, worst_fitness,
               cma.sigma, gen_secs, total_secs, best_score_ever);

        // Save best params periodically
        if ((gen + 1) % 10 == 0 || gen + 1 == generations) {
            SimParams best_p = vec_to_params(best_vec_ever);
            save_params(output, best_p);

            // Print current best params
            printf("  Current best parameters:\n");
            for (int j = 0; j < N_PARAMS; j++) {
                printf("    %-25s = %.4f  (default %.4f, range [%.3f, %.3f])\n",
                       PARAM_DEFS[j].name, best_vec_ever[j], PARAM_DEFS[j].default_val,
                       PARAM_DEFS[j].lo, PARAM_DEFS[j].hi);
            }
        }
    }

    // Final validation with more rollouts on all cases
    printf("\n=== Final Validation (100 rollouts, all cases) ===\n");
    SimParams best_params = vec_to_params(best_vec_ever);
    double total_score = 0;
    int n_validated = 0;
    for (auto& tc : all_cases) {
        double wkl = evaluate_params(tc.grid, tc.gt, best_params, 100);
        double score = std::max(0.0, std::min(100.0, 100.0 * exp(-3.0 * wkl)));
        printf("  R%d.S%d: %.2f\n", tc.round, tc.seed, score);
        total_score += score;
        n_validated++;
    }
    printf("  Average: %.2f (across %d cases)\n", total_score / n_validated, n_validated);

    // Save final
    save_params(output, best_params);
    printf("\nDone! Best params saved to %s\n", output.c_str());

    return 0;
}
