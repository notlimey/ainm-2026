// tune.cpp — Per-round parameter tuning using query observations
// Uses Nelder-Mead to optimize key sim params against observed query data.
// Usage: ./tune data/grids.bin <round> --params data/params.bin --queries-dir ../aggregate-data/data/queries
//        [--output data/params_tuned.bin] [--rollouts 30] [--iterations 80]

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <vector>
#include <string>
#include <fstream>
#include <sstream>
#include <algorithm>
#include <random>
#include <numeric>
#include <set>
#include "database.hpp"
#include "features.hpp"
#include "sim_params.hpp"

// ─── Minimal JSON parser for query files ────────────────────────────────────
// We only need: viewport.{x,y,w,h} and grid[][]

struct QueryObs {
    int vx, vy, vw, vh;
    std::vector<std::vector<int>> grid; // vh x vw
};

// Simple JSON number extraction
static int json_int_after(const std::string& s, const std::string& key, size_t start = 0) {
    auto pos = s.find("\"" + key + "\"", start);
    if (pos == std::string::npos) return -1;
    pos = s.find(':', pos);
    if (pos == std::string::npos) return -1;
    pos++;
    while (pos < s.size() && (s[pos] == ' ' || s[pos] == '\n' || s[pos] == '\r')) pos++;
    return atoi(s.c_str() + pos);
}

std::vector<QueryObs> load_queries(const std::string& path) {
    std::vector<QueryObs> result;
    std::ifstream f(path);
    if (!f) return result;
    std::string content((std::istreambuf_iterator<char>(f)),
                         std::istreambuf_iterator<char>());

    // Split by query objects — find each "query_index" occurrence
    size_t pos = 0;
    while (true) {
        auto qi = content.find("\"query_index\"", pos);
        if (qi == std::string::npos) break;

        // Find the viewport
        auto vp = content.find("\"viewport\"", qi);
        if (vp == std::string::npos) break;

        QueryObs q;
        q.vx = json_int_after(content, "x", vp);
        q.vy = json_int_after(content, "y", vp + 10); // skip past "x"
        q.vw = json_int_after(content, "w", vp);
        q.vh = json_int_after(content, "h", vp);

        if (q.vx < 0 || q.vy < 0 || q.vw <= 0 || q.vh <= 0) { pos = qi + 1; continue; }

        // Find "grid" array after viewport
        auto gp = content.find("\"grid\"", vp);
        if (gp == std::string::npos) break;

        // Parse grid: find opening '[' then parse row by row
        auto arr_start = content.find('[', gp + 5);
        if (arr_start == std::string::npos) break;

        q.grid.resize(q.vh);
        size_t scan = arr_start + 1; // skip outer '['
        for (int r = 0; r < q.vh; r++) {
            q.grid[r].resize(q.vw);
            auto row_start = content.find('[', scan);
            if (row_start == std::string::npos) break;
            scan = row_start + 1;
            for (int c = 0; c < q.vw; c++) {
                while (scan < content.size() && !isdigit(content[scan]) && content[scan] != '-') scan++;
                q.grid[r][c] = atoi(content.c_str() + scan);
                while (scan < content.size() && (isdigit(content[scan]) || content[scan] == '-')) scan++;
                while (scan < content.size() && (content[scan] == ',' || content[scan] == ' ')) scan++;
            }
            auto row_end = content.find(']', scan);
            if (row_end != std::string::npos) scan = row_end + 1;
        }

        result.push_back(q);
        pos = qi + 1;
    }
    return result;
}

// ─── Accumulate query observations into per-cell class counts ───────────────

struct ObsData {
    int W, H;
    std::vector<std::vector<std::vector<int>>> counts;   // [H][W][6]
    std::vector<std::vector<int>> sample_count;           // [H][W]
    int total_observed_cells = 0;

    void init(int w, int h) {
        W = w; H = h;
        counts.assign(h, std::vector<std::vector<int>>(w, std::vector<int>(NUM_CLASSES, 0)));
        sample_count.assign(h, std::vector<int>(w, 0));
    }

    void add_query(const QueryObs& q) {
        for (int r = 0; r < q.vh && (q.vy + r) < H; r++) {
            for (int c = 0; c < q.vw && (q.vx + c) < W; c++) {
                int gy = q.vy + r, gx = q.vx + c;
                int terrain = q.grid[r][c];
                int cls = terrain_to_class(terrain);
                counts[gy][gx][cls]++;
                sample_count[gy][gx]++;
            }
        }
    }

    void finalize() {
        total_observed_cells = 0;
        for (int y = 0; y < H; y++)
            for (int x = 0; x < W; x++)
                if (sample_count[y][x] > 0) total_observed_cells++;
    }
};

// ─── Simulation code (self-contained for speed) ─────────────────────────────

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
    float randn(float m, float s) { return std::normal_distribution<float>(m, s)(rng); }
};

SimWorld init_world(const std::vector<std::vector<int>>& ig, const SimParams& p, uint32_t seed) {
    SimWorld w;
    w.H = ig.size(); w.W = w.H > 0 ? ig[0].size() : 0;
    w.rng.seed(seed); w.grid = ig;
    w.settlements.reserve(w.W * w.H);
    int nxt = 0;
    for (int y = 0; y < w.H; y++) for (int x = 0; x < w.W; x++) {
        int t = w.grid[y][x];
        if (t == TERRAIN_SETTLEMENT || t == TERRAIN_PORT) {
            SimSettlement s;
            s.x = x; s.y = y;
            s.population = p.init_population * (0.8f + w.randf() * 0.4f);
            s.food = p.init_food * (0.8f + w.randf() * 0.4f);
            s.wealth = 0; s.defense = p.init_defense * (0.8f + w.randf() * 0.4f);
            s.tech = p.init_tech; s.has_port = (t == TERRAIN_PORT);
            s.alive = true; s.owner_id = nxt++; s.longships = s.has_port ? 1 : 0;
            w.settlements.push_back(s);
        }
    }
    w.rebuild_settle_map();
    return w;
}

void phase_growth(SimWorld& w, const SimParams& p) {
    struct EC { int pi, tx, ty; };
    std::vector<EC> exps;
    for (int i = 0; i < (int)w.settlements.size(); i++) {
        auto& s = w.settlements[i];
        if (!s.alive) continue;
        float fg = 0;
        for (int d = 0; d < 8; d++) {
            int nx = s.x+DX[d], ny = s.y+DY[d];
            if (nx<0||nx>=w.W||ny<0||ny>=w.H) continue;
            int t = w.grid[ny][nx];
            if (t == TERRAIN_FOREST) fg += p.food_per_forest;
            else if (t == TERRAIN_PLAINS || t == TERRAIN_EMPTY) fg += p.food_per_plains;
        }
        if (w.is_coastal(s.x, s.y)) fg += p.food_per_coastal;
        fg *= (1.0f + 0.1f * s.tech);
        s.food += fg; s.food = std::min(s.food, 1.0f);
        if (s.food > p.growth_threshold) {
            float g = p.growth_rate * s.food; s.population += g; s.food -= g * 0.4f;
        }
        if (s.defense < 1.0f) s.defense += 0.02f;
        s.defense = std::min(s.defense, 1.0f);
        if (!s.has_port && s.population > p.port_threshold && w.is_coastal(s.x, s.y))
            if (w.randf() < p.port_prob) { s.has_port = true; w.grid[s.y][s.x] = TERRAIN_PORT; }
        if (s.has_port && s.longships < 2 && s.population > p.longship_threshold)
            if (w.randf() < p.longship_prob) s.longships++;
        if (s.population > p.expansion_pop && w.randf() < p.expansion_prob) {
            std::vector<std::pair<int,int>> cands;
            std::vector<float> wts;
            int r = p.expansion_range;
            for (int dy = -r; dy <= r; dy++) for (int dx = -r; dx <= r; dx++) {
                if (!dx && !dy) continue;
                int tx = s.x+dx, ty = s.y+dy;
                if (tx<0||tx>=w.W||ty<0||ty>=w.H) continue;
                int t = w.grid[ty][tx];
                if ((t==TERRAIN_PLAINS||t==TERRAIN_EMPTY||t==TERRAIN_FOREST) && w.settle_map[ty][tx]==-1) {
                    cands.push_back({tx,ty});
                    float dist = sqrtf((float)(dx*dx+dy*dy));
                    float tw = (t==TERRAIN_FOREST) ? 0.5f : 1.0f;
                    wts.push_back(tw / std::max(dist, 1.0f));
                }
            }
            if (!cands.empty()) {
                float tot = 0; for (float v : wts) tot += v;
                float pick = w.randf() * tot;
                int ch = 0;
                for (int ci = 0; ci < (int)wts.size(); ci++) { pick -= wts[ci]; if (pick <= 0) { ch = ci; break; } }
                exps.push_back({i, cands[ch].first, cands[ch].second});
            }
        }
    }
    for (auto& e : exps) {
        auto& par = w.settlements[e.pi];
        if (!par.alive || par.population < p.expansion_pop * 0.5f) continue;
        if (w.settle_map[e.ty][e.tx] != -1) continue;
        SimSettlement ns;
        ns.x = e.tx; ns.y = e.ty;
        ns.population = par.population * 0.25f; par.population -= ns.population;
        ns.food = 0.3f; ns.wealth = 0; ns.defense = 0.5f;
        ns.tech = par.tech * 0.5f; ns.has_port = false; ns.alive = true;
        ns.owner_id = par.owner_id; ns.longships = 0;
        if (w.is_coastal(e.tx, e.ty) && w.randf() < 0.15f) {
            ns.has_port = true; w.grid[e.ty][e.tx] = TERRAIN_PORT;
        } else { w.grid[e.ty][e.tx] = TERRAIN_SETTLEMENT; }
        w.settle_map[e.ty][e.tx] = (int)w.settlements.size();
        w.settlements.push_back(ns);
    }
    if (!exps.empty()) w.rebuild_settle_map();
}

void phase_conflict(SimWorld& w, const SimParams& p) {
    std::vector<int> ord;
    for (int i = 0; i < (int)w.settlements.size(); i++)
        if (w.settlements[i].alive) ord.push_back(i);
    std::shuffle(ord.begin(), ord.end(), w.rng);
    for (int idx : ord) {
        auto& s = w.settlements[idx];
        if (!s.alive) continue;
        float rp = p.raid_prob_base;
        if (s.food < p.desperation_food) rp = p.raid_prob_desperate;
        if (w.randf() > rp) continue;
        float range = (s.longships > 0) ? p.raid_range_sea : p.raid_range_land;
        std::vector<int> tgts;
        for (int j = 0; j < (int)w.settlements.size(); j++) {
            if (j==idx || !w.settlements[j].alive || w.settlements[j].owner_id==s.owner_id) continue;
            float d = sqrtf((float)((s.x-w.settlements[j].x)*(s.x-w.settlements[j].x)+
                                    (s.y-w.settlements[j].y)*(s.y-w.settlements[j].y)));
            if (d <= range) tgts.push_back(j);
        }
        if (tgts.empty()) continue;
        int ti = tgts[w.rng() % tgts.size()];
        auto& tgt = w.settlements[ti];
        w.mark_war(s.owner_id, tgt.owner_id);
        float atk = s.population * (1.0f + 0.2f * s.tech);
        float def = tgt.defense * tgt.population;
        if (atk > def * (0.5f + w.randf())) {
            float loot = std::min(tgt.food * p.raid_loot_frac, 0.2f);
            s.food += loot; tgt.food -= loot;
            s.wealth += 0.01f; tgt.defense -= p.raid_damage; tgt.population -= 0.05f;
            if (tgt.defense < 0.1f && w.randf() < p.conquest_prob) tgt.owner_id = s.owner_id;
        }
    }
}

void phase_trade(SimWorld& w, const SimParams& p) {
    for (int i = 0; i < (int)w.settlements.size(); i++) {
        auto& a = w.settlements[i];
        if (!a.alive || !a.has_port) continue;
        for (int j = i+1; j < (int)w.settlements.size(); j++) {
            auto& b = w.settlements[j];
            if (!b.alive || !b.has_port) continue;
            if (w.at_war(a.owner_id, b.owner_id)) continue;
            float d = sqrtf((float)((a.x-b.x)*(a.x-b.x)+(a.y-b.y)*(a.y-b.y)));
            if (d > p.trade_range) continue;
            a.food += p.trade_food; b.food += p.trade_food;
            a.wealth += p.trade_wealth; b.wealth += p.trade_wealth;
            float avg = (a.tech + b.tech) * 0.5f;
            a.tech += (avg - a.tech) * p.tech_diffusion;
            b.tech += (avg - b.tech) * p.tech_diffusion;
        }
    }
}

void phase_winter(SimWorld& w, const SimParams& p) {
    float sev = p.winter_base_loss + w.randn(0, p.winter_variance);
    if (w.randf() < p.winter_catastrophe_prob) sev *= p.winter_catastrophe_mult;
    sev = std::max(0.05f, std::min(sev, 0.9f));
    std::vector<int> col;
    for (int i = 0; i < (int)w.settlements.size(); i++) {
        auto& s = w.settlements[i];
        if (!s.alive) continue;
        s.food -= sev * (1.0f + 0.05f * s.population);
        if (s.food < 0) { s.population += s.food * 0.3f; if (s.population < 0) s.population = 0; s.food = 0; }
        if (s.population < p.collapse_pop || (s.food < p.collapse_food + 0.01f && s.defense < p.collapse_defense))
            col.push_back(i);
    }
    for (int idx : col) {
        auto& s = w.settlements[idx];
        s.alive = false; w.grid[s.y][s.x] = TERRAIN_RUIN; w.settle_map[s.y][s.x] = -1;
        float pop = s.population;
        std::vector<int> near;
        for (int j = 0; j < (int)w.settlements.size(); j++) {
            if (j==idx || !w.settlements[j].alive || w.settlements[j].owner_id!=s.owner_id) continue;
            float d = sqrtf((float)((s.x-w.settlements[j].x)*(s.x-w.settlements[j].x)+
                                    (s.y-w.settlements[j].y)*(s.y-w.settlements[j].y)));
            if (d <= 5.0f) near.push_back(j);
        }
        if (!near.empty()) { float sh = pop/near.size(); for (int j : near) w.settlements[j].population += sh; }
    }
}

void phase_environment(SimWorld& w, const SimParams& p) {
    std::vector<std::tuple<int,int,int>> ch;
    for (int y = 0; y < w.H; y++) for (int x = 0; x < w.W; x++) {
        if (w.grid[y][x] != TERRAIN_RUIN) continue;
        int bi = -1; float bd = 999;
        for (int i = 0; i < (int)w.settlements.size(); i++) {
            auto& s = w.settlements[i]; if (!s.alive) continue;
            float d = sqrtf((float)((x-s.x)*(x-s.x)+(y-s.y)*(y-s.y)));
            if (d <= p.ruin_reclaim_range && d < bd) { bd = d; bi = i; }
        }
        if (bi >= 0 && w.randf() < p.ruin_reclaim_prob) {
            auto& pat = w.settlements[bi];
            SimSettlement ns; ns.x=x; ns.y=y;
            ns.population = pat.population*0.15f; ns.food=0.2f; ns.wealth=0; ns.defense=0.4f;
            ns.tech = pat.tech*0.3f; ns.alive=true; ns.owner_id=pat.owner_id; ns.longships=0;
            if (w.is_coastal(x,y) && w.randf()<0.3f) { ns.has_port=true; ch.push_back({x,y,TERRAIN_PORT}); }
            else { ns.has_port=false; ch.push_back({x,y,TERRAIN_SETTLEMENT}); }
            w.settlements.push_back(ns); continue;
        }
        int af = w.count_adjacent(x,y,TERRAIN_FOREST);
        float fc = p.ruin_forest_prob + af * p.forest_adj_bonus;
        if (w.randf() < fc) { ch.push_back({x,y,TERRAIN_FOREST}); continue; }
        if (w.randf() < p.ruin_plains_prob) ch.push_back({x,y,TERRAIN_PLAINS});
    }
    for (auto& [x,y,t] : ch) w.grid[y][x] = t;
    w.rebuild_settle_map();
}

void simulate_one(SimWorld& w, const SimParams& p) {
    for (int yr = 0; yr < 50; yr++) {
        w.war_pairs.clear();
        phase_growth(w, p);
        phase_conflict(w, p);
        phase_trade(w, p);
        phase_winter(w, p);
        phase_environment(w, p);
        if ((int)w.settlements.size() > w.W * w.H * 2) break;
    }
}

// ─── Jitter (same as simulate.cpp) ──────────────────────────────────────────

SimParams jitter_params(const SimParams& base, std::mt19937& rng, float sc = 0.08f) {
    SimParams p = base;
    auto j = [&](float v, float lo, float hi) -> float {
        float n = std::normal_distribution<float>(0, sc*(hi-lo))(rng);
        return std::max(lo, std::min(hi, v+n));
    };
    p.expansion_pop = j(p.expansion_pop,0.5f,4.0f);
    p.expansion_prob = j(p.expansion_prob,0.05f,0.8f);
    p.growth_rate = j(p.growth_rate,0.02f,0.25f);
    p.growth_threshold = j(p.growth_threshold,0.1f,0.8f);
    p.food_per_forest = j(p.food_per_forest,0.02f,0.4f);
    p.food_per_plains = j(p.food_per_plains,0.01f,0.15f);
    p.raid_prob_base = j(p.raid_prob_base,0.01f,0.25f);
    p.raid_damage = j(p.raid_damage,0.05f,0.6f);
    p.conquest_prob = j(p.conquest_prob,0.02f,0.35f);
    p.winter_base_loss = j(p.winter_base_loss,0.05f,0.6f);
    p.winter_variance = j(p.winter_variance,0.02f,0.3f);
    p.collapse_pop = j(p.collapse_pop,0.01f,0.3f);
    p.ruin_reclaim_prob = j(p.ruin_reclaim_prob,0.05f,0.5f);
    p.ruin_forest_prob = j(p.ruin_forest_prob,0.03f,0.4f);
    { float er = j((float)p.expansion_range,1,8); p.expansion_range = std::max(1,std::min(8,(int)roundf(er))); }
    return p;
}

// ─── Scoring: KL on observed cells only ─────────────────────────────────────

float evaluate_params(
    const SimParams& params,
    const std::vector<std::vector<int>>& initial_grid,
    const ObsData& obs,
    int num_rollouts,
    int num_seeds
) {
    int H = obs.H, W = obs.W;

    // Run sim with multiple seeds, accumulate counts
    std::vector<std::vector<std::vector<int>>> counts(
        H, std::vector<std::vector<int>>(W, std::vector<int>(NUM_CLASSES, 0)));
    int total_rollouts = 0;

    std::mt19937 jitter_rng(12345);
    for (int s = 0; s < num_seeds; s++) {
        for (int r = 0; r < num_rollouts; r++) {
            SimParams jp = jitter_params(params, jitter_rng);
            SimWorld world = init_world(initial_grid, jp, 42 + s * 100000 + r * 7919);
            simulate_one(world, jp);
            for (int y = 0; y < H; y++)
                for (int x = 0; x < W; x++)
                    counts[y][x][terrain_to_class(world.grid[y][x])]++;
            total_rollouts++;
        }
    }

    // Compute KL only on observed cells, weighted by entropy
    double total_wkl = 0, total_entropy = 0;

    for (int y = 0; y < H; y++) {
        for (int x = 0; x < W; x++) {
            if (obs.sample_count[y][x] < 2) continue; // need at least 2 observations

            // Observed distribution (from queries)
            float obs_probs[NUM_CLASSES];
            float obs_total = (float)obs.sample_count[y][x];
            for (int c = 0; c < NUM_CLASSES; c++)
                obs_probs[c] = (float)obs.counts[y][x][c] / obs_total;

            // Entropy of observed distribution
            double ent = 0;
            for (int c = 0; c < NUM_CLASSES; c++)
                if (obs_probs[c] > 1e-10) ent -= obs_probs[c] * log(obs_probs[c]);
            if (ent < 1e-6) continue; // skip static cells

            // Sim distribution
            int init_terrain = initial_grid[y][x];
            bool reachable[NUM_CLASSES] = {};
            if (init_terrain == TERRAIN_MOUNTAIN) { reachable[5] = true; }
            else if (init_terrain == TERRAIN_OCEAN) { reachable[0] = true; }
            else { for (int c = 0; c < 5; c++) reachable[c] = true; }

            float sim_probs[NUM_CLASSES];
            float sim_total = 0;
            for (int c = 0; c < NUM_CLASSES; c++) {
                float raw = (float)counts[y][x][c] / total_rollouts;
                sim_probs[c] = reachable[c] ? std::max(raw, 0.003f) : 0.0f;
                sim_total += sim_probs[c];
            }
            if (sim_total > 0) for (int c = 0; c < NUM_CLASSES; c++) sim_probs[c] /= sim_total;

            // KL(obs || sim)
            double kl = 0;
            for (int c = 0; c < NUM_CLASSES; c++) {
                if (obs_probs[c] > 1e-10)
                    kl += obs_probs[c] * log(obs_probs[c] / std::max(sim_probs[c], 1e-10f));
            }

            total_wkl += ent * kl;
            total_entropy += ent;
        }
    }

    if (total_entropy < 1e-10) return 100.0f;
    double wkl = total_wkl / total_entropy;
    return std::max(0.0, std::min(100.0, 100.0 * exp(-3.0 * wkl)));
}

// ─── Nelder-Mead on key parameters ─────────────────────────────────────────

// Which params to tune (indices into a float vector)
struct TuneParam {
    const char* name;
    float* ptr;  // pointer into SimParams
    float lo, hi;
};

void get_tune_params(SimParams& p, std::vector<TuneParam>& tp) {
    tp = {
        {"expansion_pop",   &p.expansion_pop,    0.5f, 4.0f},
        {"expansion_prob",  &p.expansion_prob,    0.02f, 0.6f},
        {"growth_rate",     &p.growth_rate,       0.01f, 0.3f},
        {"raid_prob_base",  &p.raid_prob_base,    0.01f, 0.3f},
        {"winter_base_loss",&p.winter_base_loss,  0.05f, 0.8f},
        {"collapse_pop",    &p.collapse_pop,      0.01f, 0.5f},
        {"ruin_reclaim_prob",&p.ruin_reclaim_prob, 0.02f, 0.5f},
        {"food_per_forest", &p.food_per_forest,   0.02f, 0.5f},
    };
}

int main(int argc, char* argv[]) {
    if (argc < 3) {
        printf("Usage: tune <grids.bin> <round> --params <params.bin> --queries-dir <dir>\n");
        printf("       [--output <params_tuned.bin>] [--rollouts 30] [--iterations 80]\n");
        return 1;
    }

    std::string grids_path = argv[1];
    int round = atoi(argv[2]);
    std::string params_path = "data/params.bin";
    std::string queries_dir = "../aggregate-data/data/queries";
    std::string output_path = "data/params_tuned.bin";
    int rollouts = 30;
    int iterations = 80;

    for (int i = 3; i < argc; i++) {
        if (!strcmp(argv[i], "--params") && i+1 < argc) params_path = argv[++i];
        else if (!strcmp(argv[i], "--queries-dir") && i+1 < argc) queries_dir = argv[++i];
        else if (!strcmp(argv[i], "--output") && i+1 < argc) output_path = argv[++i];
        else if (!strcmp(argv[i], "--rollouts") && i+1 < argc) rollouts = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--iterations") && i+1 < argc) iterations = atoi(argv[++i]);
    }

    printf("Per-Round Parameter Tuning\n");
    printf("  Round:      %d\n", round);
    printf("  Rollouts:   %d\n", rollouts);
    printf("  Iterations: %d\n", iterations);

    // Load base params
    SimParams base_params;
    if (!load_params(params_path, base_params)) {
        printf("Warning: could not load params, using defaults\n");
    }

    // Load initial grids for all seeds of this round
    // Load from grids.bin — find all seeds for this round
    struct SeedData {
        int seed;
        std::vector<std::vector<int>> grid;
        ObsData obs;
    };
    std::vector<SeedData> seeds;

    // Load grids
    FILE* gf = fopen(grids_path.c_str(), "rb");
    if (!gf) { printf("Cannot open grids: %s\n", grids_path.c_str()); return 1; }
    char magic[4]; fread(magic, 1, 4, gf);
    uint16_t ver; fread(&ver, 2, 1, gf);
    uint32_t count; fread(&count, 4, 1, gf);
    for (uint32_t i = 0; i < count; i++) {
        uint32_t r, s, w, h;
        fread(&r, 4, 1, gf); fread(&s, 4, 1, gf);
        fread(&w, 4, 1, gf); fread(&h, 4, 1, gf);
        std::vector<std::vector<int>> grid(h, std::vector<int>(w));
        for (uint32_t y = 0; y < h; y++)
            for (uint32_t x = 0; x < w; x++) {
                int32_t val; fread(&val, 4, 1, gf);
                grid[y][x] = val;
            }
        if ((int)r == round) {
            SeedData sd;
            sd.seed = s;
            sd.grid = grid;
            sd.obs.init(w, h);
            seeds.push_back(sd);
        }
    }
    fclose(gf);
    printf("  Seeds:      %d\n", (int)seeds.size());

    // Load query observations
    int total_queries = 0;
    for (auto& sd : seeds) {
        char path[512];
        snprintf(path, sizeof(path), "%s/r%d/s%d_queries.json", queries_dir.c_str(), round, sd.seed);
        auto queries = load_queries(path);
        for (auto& q : queries) sd.obs.add_query(q);
        sd.obs.finalize();
        total_queries += (int)queries.size();
        printf("  Seed %d: %d queries, %d observed cells\n",
               sd.seed, (int)queries.size(), sd.obs.total_observed_cells);
    }

    if (total_queries == 0) {
        printf("No query data found! Cannot tune.\n");
        // Just copy base params
        save_params(output_path, base_params);
        return 0;
    }

    // Evaluate baseline
    auto eval_all = [&](SimParams& p) -> float {
        float total = 0;
        for (auto& sd : seeds) {
            total += evaluate_params(p, sd.grid, sd.obs, rollouts, 1);
        }
        return total / seeds.size();
    };

    float baseline_score = eval_all(base_params);
    printf("\nBaseline score (on observed cells): %.2f\n", baseline_score);

    // ─── Nelder-Mead optimization ───────────────────────────────────────────

    SimParams best = base_params;
    std::vector<TuneParam> tp;
    get_tune_params(best, tp);
    int N = tp.size();

    // Extract current values
    std::vector<float> x0(N);
    for (int i = 0; i < N; i++) x0[i] = *tp[i].ptr;

    // Create simplex: N+1 vertices
    std::vector<std::vector<float>> simplex(N + 1, x0);
    std::vector<float> scores(N + 1);

    // Perturb each dimension
    for (int i = 0; i < N; i++) {
        float range = tp[i].hi - tp[i].lo;
        simplex[i + 1][i] += range * 0.15f; // 15% perturbation
        simplex[i + 1][i] = std::min(simplex[i + 1][i], tp[i].hi);
    }

    // Evaluate initial simplex
    auto eval_vertex = [&](const std::vector<float>& v) -> float {
        SimParams p = base_params;
        std::vector<TuneParam> tp2;
        get_tune_params(p, tp2);
        for (int i = 0; i < N; i++) {
            *tp2[i].ptr = std::max(tp2[i].lo, std::min(tp2[i].hi, v[i]));
        }
        return eval_all(p);
    };

    printf("Evaluating initial simplex...\n");
    for (int i = 0; i <= N; i++) {
        scores[i] = eval_vertex(simplex[i]);
        printf("\r  vertex %d/%d: %.2f", i + 1, N + 1, scores[i]);
        fflush(stdout);
    }
    printf("\n");

    // Nelder-Mead main loop
    float alpha = 1.0f, gamma = 2.0f, rho = 0.5f, sigma = 0.5f;

    for (int iter = 0; iter < iterations; iter++) {
        // Sort vertices by score (descending — we maximize)
        std::vector<int> order(N + 1);
        std::iota(order.begin(), order.end(), 0);
        std::sort(order.begin(), order.end(), [&](int a, int b) { return scores[a] > scores[b]; });

        int best_idx = order[0], worst_idx = order[N], second_worst_idx = order[N - 1];
        float f_best = scores[best_idx], f_worst = scores[worst_idx], f_second_worst = scores[second_worst_idx];

        // Centroid (excluding worst)
        std::vector<float> centroid(N, 0);
        for (int i = 0; i < N; i++) { // first N vertices (excluding worst)
            for (int d = 0; d < N; d++)
                centroid[d] += simplex[order[i]][d];
        }
        for (int d = 0; d < N; d++) centroid[d] /= N;

        // Reflection
        std::vector<float> reflected(N);
        for (int d = 0; d < N; d++)
            reflected[d] = centroid[d] + alpha * (centroid[d] - simplex[worst_idx][d]);
        float f_reflected = eval_vertex(reflected);

        if (f_reflected > f_second_worst && f_reflected <= f_best) {
            // Accept reflection
            simplex[worst_idx] = reflected;
            scores[worst_idx] = f_reflected;
        } else if (f_reflected > f_best) {
            // Expansion
            std::vector<float> expanded(N);
            for (int d = 0; d < N; d++)
                expanded[d] = centroid[d] + gamma * (reflected[d] - centroid[d]);
            float f_expanded = eval_vertex(expanded);
            if (f_expanded > f_reflected) {
                simplex[worst_idx] = expanded;
                scores[worst_idx] = f_expanded;
            } else {
                simplex[worst_idx] = reflected;
                scores[worst_idx] = f_reflected;
            }
        } else {
            // Contraction
            std::vector<float> contracted(N);
            for (int d = 0; d < N; d++)
                contracted[d] = centroid[d] + rho * (simplex[worst_idx][d] - centroid[d]);
            float f_contracted = eval_vertex(contracted);
            if (f_contracted > f_worst) {
                simplex[worst_idx] = contracted;
                scores[worst_idx] = f_contracted;
            } else {
                // Shrink
                for (int i = 1; i <= N; i++) {
                    for (int d = 0; d < N; d++)
                        simplex[order[i]][d] = simplex[best_idx][d] +
                            sigma * (simplex[order[i]][d] - simplex[best_idx][d]);
                    scores[order[i]] = eval_vertex(simplex[order[i]]);
                }
            }
        }

        // Find current best
        float cur_best = *std::max_element(scores.begin(), scores.end());
        if ((iter + 1) % 5 == 0 || iter == iterations - 1)
            printf("  iter %3d/%d: best=%.2f (baseline=%.2f, +%.2f)\n",
                   iter + 1, iterations, cur_best, baseline_score, cur_best - baseline_score);
    }

    // Extract best vertex
    int final_best = std::max_element(scores.begin(), scores.end()) - scores.begin();
    SimParams tuned = base_params;
    std::vector<TuneParam> tp_final;
    get_tune_params(tuned, tp_final);
    for (int i = 0; i < N; i++)
        *tp_final[i].ptr = std::max(tp_final[i].lo, std::min(tp_final[i].hi, simplex[final_best][i]));

    float final_score = eval_all(tuned);
    printf("\nFinal score: %.2f (baseline: %.2f, improvement: %+.2f)\n",
           final_score, baseline_score, final_score - baseline_score);

    printf("\nTuned parameters:\n");
    for (int i = 0; i < N; i++) {
        printf("  %-20s: %.4f -> %.4f\n", tp_final[i].name, x0[i], *tp_final[i].ptr);
    }

    save_params(output_path, tuned);
    printf("\nSaved to: %s\n", output_path.c_str());

    return 0;
}
