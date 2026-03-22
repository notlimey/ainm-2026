#pragma once
// Shared I/O and scoring utilities for ASTP/grid/ground-truth binary formats.
// Used by: simulate.cpp, predict.cpp, mlp.cpp, cnn.cpp
// blend.cpp uses flat ProbGrid and has its own I/O.

#include <cstdio>
#include <cstdint>
#include <cmath>
#include <string>
#include <vector>
#include <fstream>
#include <algorithm>
#include "database.hpp"
// NUM_CLASSES must be defined before including this header (via features.hpp)

// ─── Binary grid loading ─────────────────────────────────────────────────────

inline bool load_grid(const std::string& path, int want_round, int want_seed,
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

// ─── Ground truth loading ────────────────────────────────────────────────────

inline bool load_ground_truth(const std::string& path, int want_round, int want_seed,
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

// ─── ASTP prediction writing ─────────────────────────────────────────────────

inline void write_prediction(const std::string& path, int round, int seed, int W, int H,
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

// ─── Probability floor + renormalize ─────────────────────────────────────────

inline void apply_floor(float* probs, float floor_val = 0.005f) {
    for (int c = 0; c < NUM_CLASSES; c++)
        if (probs[c] < floor_val) probs[c] = floor_val;
    float total = 0;
    for (int c = 0; c < NUM_CLASSES; c++) total += probs[c];
    if (total > 0) for (int c = 0; c < NUM_CLASSES; c++) probs[c] /= total;
}

// ─── Scoring (entropy-weighted KL divergence) ────────────────────────────────

inline float score_prediction(const std::vector<std::vector<std::vector<float>>>& gt,
                              const std::vector<std::vector<std::vector<float>>>& pred,
                              int W, int H) {
    double total_entropy = 0, total_weighted_kl = 0;
    int dynamic_cells = 0;

    for (int y = 0; y < H; y++) {
        for (int x = 0; x < W; x++) {
            double ent = 0;
            for (int c = 0; c < NUM_CLASSES; c++)
                if (gt[y][x][c] > 0) ent -= gt[y][x][c] * log(gt[y][x][c]);
            if (ent < 1e-6) continue;
            dynamic_cells++;

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
