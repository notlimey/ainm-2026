#pragma once
// sim_core.hpp — Shared simulation logic used by simulate.cpp, tune.cpp, and calibrate.cpp.
// Single source of truth for: SimSettlement, SimWorld, init_world, all 5 phases, simulate_one.
// Any changes to simulation mechanics MUST be made here.

#include <cmath>
#include <vector>
#include <algorithm>
#include <random>
#include <set>
#include "database.hpp"
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

inline SimWorld init_world(const std::vector<std::vector<int>>& initial_grid,
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

inline void phase_growth(SimWorld& w, const SimParams& p) {
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
        food_gain *= (1.0f + p.tech_food_bonus * s.tech);
        s.food += food_gain;
        s.food = std::min(s.food, p.food_cap);

        // Population growth (wealth accelerates growth)
        if (s.food > p.growth_threshold) {
            float growth = p.growth_rate * s.food * (1.0f + p.wealth_growth_bonus * s.wealth);
            s.population += growth;
            s.food -= growth * p.growth_food_cost;
        }

        // Defense recovery (wealth accelerates recovery)
        if (s.defense < 1.0f) s.defense += p.defense_recovery * (1.0f + p.wealth_defense_bonus * s.wealth);
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
            std::vector<std::pair<int,int>> candidates;
            std::vector<float> weights;
            int r = p.expansion_range;
            for (int dy = -r; dy <= r; dy++) {
                for (int dx = -r; dx <= r; dx++) {
                    if (dx == 0 && dy == 0) continue;
                    int tx = s.x + dx, ty = s.y + dy;
                    if (tx < 0 || tx >= w.W || ty < 0 || ty >= w.H) continue;
                    int t = w.grid[ty][tx];
                    if (t == TERRAIN_PLAINS || t == TERRAIN_EMPTY || t == TERRAIN_FOREST) {
                        if (w.settle_map[ty][tx] == -1) {
                            candidates.push_back({tx, ty});
                            float dist = sqrtf((float)(dx*dx + dy*dy));
                            float terrain_w = (t == TERRAIN_FOREST) ? 0.5f : 1.0f;
                            weights.push_back(terrain_w / std::max(dist, 1.0f));
                        }
                    }
                }
            }
            if (!candidates.empty()) {
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

    // Apply expansions
    for (auto& e : expansions) {
        auto& parent = w.settlements[e.parent_idx];
        if (!parent.alive || parent.population < p.expansion_pop * 0.5f) continue;
        if (w.settle_map[e.ty][e.tx] != -1) continue;

        SimSettlement ns;
        ns.x = e.tx; ns.y = e.ty;
        ns.population = parent.population * p.expansion_split;
        parent.population -= ns.population;
        ns.food = p.new_settle_food;
        ns.wealth = 0.0f;
        ns.defense = p.new_settle_defense;
        ns.tech = parent.tech * p.new_settle_tech_frac;
        ns.has_port = false;
        ns.alive = true;
        ns.owner_id = parent.owner_id;
        ns.longships = 0;

        if (w.is_coastal(e.tx, e.ty) && w.randf() < p.new_settle_port_prob) {
            ns.has_port = true;
            w.grid[e.ty][e.tx] = TERRAIN_PORT;
        } else {
            w.grid[e.ty][e.tx] = TERRAIN_SETTLEMENT;
        }

        w.settle_map[e.ty][e.tx] = (int)w.settlements.size();
        w.settlements.push_back(ns);
    }

    if (!expansions.empty()) w.rebuild_settle_map();
}

// ─── Phase 2: Conflict ──────────────────────────────────────────────────────

inline void phase_conflict(SimWorld& w, const SimParams& p) {
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

        float attack = s.population * (1.0f + p.tech_attack_bonus * s.tech);
        float defend = target.defense * target.population;
        float roll = 0.5f + w.randf();

        if (attack > defend * roll) {
            float loot = std::min(target.food * p.raid_loot_frac, p.raid_loot_cap);
            s.food += loot;
            target.food -= loot;
            s.wealth += p.raid_wealth_gain;
            target.defense -= p.raid_damage;
            target.population -= p.raid_pop_loss;

            if (target.defense < p.conquest_threshold && w.randf() < p.conquest_prob) {
                target.owner_id = s.owner_id;
            }
        }
    }
}

// ─── Phase 3: Trade ─────────────────────────────────────────────────────────

inline void phase_trade(SimWorld& w, const SimParams& p) {
    for (int i = 0; i < (int)w.settlements.size(); i++) {
        auto& a = w.settlements[i];
        if (!a.alive || !a.has_port) continue;

        for (int j = i + 1; j < (int)w.settlements.size(); j++) {
            auto& b = w.settlements[j];
            if (!b.alive || !b.has_port) continue;
            if (w.at_war(a.owner_id, b.owner_id)) continue;

            float dist = sqrtf((a.x - b.x) * (a.x - b.x) + (a.y - b.y) * (a.y - b.y));
            if (dist > p.trade_range) continue;

            a.food += p.trade_food;
            b.food += p.trade_food;
            a.wealth += p.trade_wealth;
            b.wealth += p.trade_wealth;

            float avg_tech = (a.tech + b.tech) * 0.5f;
            a.tech += (avg_tech - a.tech) * p.tech_diffusion;
            b.tech += (avg_tech - b.tech) * p.tech_diffusion;
        }
    }
}

// ─── Phase 4: Winter ────────────────────────────────────────────────────────

inline void phase_winter(SimWorld& w, const SimParams& p) {
    float severity = p.winter_base_loss + w.randn(0.0f, p.winter_variance);
    if (w.randf() < p.winter_catastrophe_prob) {
        severity *= p.winter_catastrophe_mult;
    }
    severity = std::max(p.winter_severity_min, std::min(severity, p.winter_severity_max));

    std::vector<int> to_collapse;

    for (int i = 0; i < (int)w.settlements.size(); i++) {
        auto& s = w.settlements[i];
        if (!s.alive) continue;

        float food_loss = severity * (1.0f + p.winter_pop_scale * s.population);
        s.food -= food_loss;

        if (s.food < 0) {
            s.population += s.food * p.starvation_rate;
            if (s.population < 0) s.population = 0;
            s.food = 0;
        }

        if (s.population < p.collapse_pop ||
            (s.food < p.collapse_food + 0.01f && s.defense < p.collapse_defense)) {
            to_collapse.push_back(i);
        }
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
            if (dist <= p.collapse_dispersion) nearby.push_back(j);
        }
        if (!nearby.empty()) {
            float share = pop_to_spread / nearby.size();
            for (int j : nearby) w.settlements[j].population += share;
        }
    }
}

// ─── Phase 5: Environment ───────────────────────────────────────────────────

inline void phase_environment(SimWorld& w, const SimParams& p) {
    std::vector<std::tuple<int,int,int>> changes;

    for (int y = 0; y < w.H; y++) {
        for (int x = 0; x < w.W; x++) {
            if (w.grid[y][x] != TERRAIN_RUIN) continue;

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
                ns.population = patron.population * p.reclaim_pop_frac;
                ns.food = p.reclaim_food;
                ns.wealth = 0.0f;
                ns.defense = p.reclaim_defense;
                ns.tech = patron.tech * p.reclaim_tech_frac;
                ns.alive = true;
                ns.owner_id = patron.owner_id;
                ns.longships = 0;

                if (w.is_coastal(x, y) && w.randf() < p.reclaim_port_prob) {
                    ns.has_port = true;
                    changes.push_back({x, y, TERRAIN_PORT});
                } else {
                    ns.has_port = false;
                    changes.push_back({x, y, TERRAIN_SETTLEMENT});
                }
                w.settlements.push_back(ns);
                continue;
            }

            int adj_forests = w.count_adjacent(x, y, TERRAIN_FOREST);
            float forest_chance = p.ruin_forest_prob + adj_forests * p.forest_adj_bonus;
            if (w.randf() < forest_chance) {
                changes.push_back({x, y, TERRAIN_FOREST});
                continue;
            }

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

inline void simulate_one(SimWorld& w, const SimParams& p) {
    for (int year = 0; year < 50; year++) {
        w.war_pairs.clear();
        phase_growth(w, p);
        phase_conflict(w, p);
        phase_trade(w, p);
        phase_winter(w, p);
        phase_environment(w, p);
    }
}
