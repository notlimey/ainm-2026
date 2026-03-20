#pragma once
// Shared SimParams definition — used by simulate.cpp, calibrate.cpp, and cnn.cpp
// Any parameter changes MUST be made here to avoid silent drift between files.

#include <cstdio>
#include <cstring>
#include <fstream>

struct SimParams {
    // Initial settlement state (unknown — calibrated)
    float init_population    = 1.0f;
    float init_food          = 0.5f;
    float init_defense       = 0.8f;
    float init_tech          = 0.1f;

    // Growth phase
    float food_per_forest    = 0.15f;
    float food_per_plains    = 0.04f;
    float food_per_coastal   = 0.03f;   // fishing bonus
    float growth_threshold   = 0.4f;    // food needed to grow
    float growth_rate        = 0.08f;
    float expansion_pop      = 1.8f;    // population to found new settlement
    int   expansion_range    = 3;       // Chebyshev distance (calibrated as continuous, rounded)
    float expansion_prob     = 0.3f;    // chance per year if eligible
    float port_threshold     = 1.3f;    // population to develop port
    float port_prob          = 0.25f;
    float longship_threshold = 1.5f;
    float longship_prob      = 0.15f;

    // Conflict phase
    float raid_range_land    = 3.0f;
    float raid_range_sea     = 8.0f;
    float raid_prob_base     = 0.08f;   // base chance to raid per year
    float raid_prob_desperate = 0.4f;   // raid chance when starving
    float desperation_food   = 0.15f;   // food threshold for desperation
    float raid_damage        = 0.25f;
    float raid_loot_frac     = 0.25f;
    float conquest_prob      = 0.12f;

    // Trade phase
    float trade_range        = 6.0f;
    float trade_food         = 0.04f;
    float trade_wealth       = 0.015f;
    float tech_diffusion     = 0.08f;

    // Winter phase
    float winter_base_loss   = 0.25f;
    float winter_variance    = 0.12f;
    float winter_catastrophe_prob = 0.05f;  // heavy tail — rare harsh winters
    float winter_catastrophe_mult = 2.5f;
    float collapse_pop       = 0.08f;   // collapse if population below this
    float collapse_food      = 0.0f;    // collapse if food AND defense below thresholds
    float collapse_defense   = 0.15f;

    // Environment phase
    float ruin_reclaim_range = 2.5f;
    float ruin_reclaim_prob  = 0.25f;
    float ruin_forest_prob   = 0.15f;
    float ruin_plains_prob   = 0.08f;
    float forest_adj_bonus   = 0.05f;   // extra forest prob per adjacent forest
};

// ─── Save/Load params (SIMP binary format) ──────────────────────────────────

inline void save_params(const std::string& path, const SimParams& p) {
    std::ofstream f(path, std::ios::binary);
    f.write("SIMP", 4);
    f.write((const char*)&p, sizeof(SimParams));
    f.close();
    printf("Params saved: %s (%zu bytes)\n", path.c_str(), sizeof(SimParams));
}

inline bool load_params(const std::string& path, SimParams& p) {
    std::ifstream f(path, std::ios::binary);
    if (!f) return false;
    char magic[4]; f.read(magic, 4);
    if (strncmp(magic, "SIMP", 4) != 0) return false;
    f.read((char*)&p, sizeof(SimParams));
    printf("Params loaded: %s\n", path.c_str());
    return true;
}
