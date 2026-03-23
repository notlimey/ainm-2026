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

// Shared simulation core (SimSettlement, SimWorld, init_world, all phases, simulate_one)
#include "sim_core.hpp"

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
