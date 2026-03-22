// blend.cpp — Per-class ensemble weight optimization for combining model predictions
// Finds optimal weights for blending bucket, MLP, simulator, and CNN predictions
// using grid search over a simplex with leave-one-round-out cross-validation.
//
// Usage: blend <ground_truth.bin> [--data-dir path] [--predict round] [--output pred_blend_r{R}_s{S}.bin]
//
// Compile: c++ -std=c++17 -O3 -o blend blend.cpp

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <cstdint>
#include <vector>
#include <string>
#include <fstream>
#include <algorithm>
#include <map>
#include <set>
#include <array>
#include <functional>

#include "features.hpp"
#include "database.hpp"

// ─── Terrain-aware reachability ──────────────────────────────────────────────

// Given initial terrain code, which classes can this cell reach after 50 years?
inline bool class_reachable(int terrain, int cls)
{
    if (terrain == TERRAIN_MOUNTAIN) return cls == 5;       // mountains stay mountains
    if (terrain == TERRAIN_OCEAN)    return cls == 0;       // ocean stays ocean
    // Land cells: everything except mountain
    return cls != 5;
}

// ─── Model names ─────────────────────────────────────────────────────────────

static const int NUM_MODELS = 4;
static const char* MODEL_NAMES[NUM_MODELS] = {"bucket", "mlp", "sim", "cnn"};

// ─── Data structures ─────────────────────────────────────────────────────────

// Flat H*W*6 probability grid
using ProbGrid = std::vector<float>;

struct GTEntry
{
    int32_t round, seed, W, H;
    ProbGrid probs;  // H * W * NUM_CLASSES
};

struct PredSet
{
    // For each model, the probability grid (or empty if not available)
    ProbGrid model[NUM_MODELS];
    bool has_model[NUM_MODELS] = {};
};

// ─── File I/O ────────────────────────────────────────────────────────────────

// Load all ground truth entries from ground_truth.bin
// Format: magic(4) + ver(2) + count(4) + entries[round(4) + seed(4) + W(4) + H(4) + float[H][W][6]]
std::vector<GTEntry> load_all_ground_truth(const std::string& path)
{
    std::vector<GTEntry> entries;
    std::ifstream f(path, std::ios::binary);
    if (!f) { printf("Error: cannot open %s\n", path.c_str()); return entries; }

    char magic[4];
    f.read(magic, 4);
    (void)magic;
    uint16_t ver;
    f.read((char*)&ver, 2);
    (void)ver;
    uint32_t count;
    f.read((char*)&count, 4);

    for (uint32_t i = 0; i < count; i++)
    {
        GTEntry e;
        f.read((char*)&e.round, 4);
        f.read((char*)&e.seed, 4);
        f.read((char*)&e.W, 4);
        f.read((char*)&e.H, 4);

        int n = e.H * e.W * NUM_CLASSES;
        e.probs.resize(n);
        f.read((char*)e.probs.data(), n * sizeof(float));

        if (!f) { printf("Error reading GT entry %d\n", i); break; }
        entries.push_back(std::move(e));
    }

    return entries;
}

// Load an ASTP prediction file
// Format: "ASTP"(4) + ver(2) + round(4) + seed(4) + W(4) + H(4) + float[H][W][6]
bool load_astp(const std::string& path, int expect_W, int expect_H, ProbGrid& out)
{
    std::ifstream f(path, std::ios::binary);
    if (!f) return false;

    char magic[4];
    f.read(magic, 4);
    if (std::string(magic, 4) != "ASTP") return false;

    uint16_t ver;
    f.read((char*)&ver, 2);
    (void)ver;

    int32_t rd, sd, W, H;
    f.read((char*)&rd, 4);
    f.read((char*)&sd, 4);
    (void)rd; (void)sd;
    f.read((char*)&W, 4);
    f.read((char*)&H, 4);

    if (W != expect_W || H != expect_H) return false;

    int n = H * W * NUM_CLASSES;
    out.resize(n);
    f.read((char*)out.data(), n * sizeof(float));

    return f.good() || f.eof();
}

// Write an ASTP prediction file
void write_astp(const std::string& path, int round, int seed, int W, int H,
                const ProbGrid& probs)
{
    std::ofstream f(path, std::ios::binary);
    f.write("ASTP", 4);
    uint16_t ver = 1;
    f.write((char*)&ver, 2);
    f.write((char*)&round, 4);
    f.write((char*)&seed, 4);
    f.write((char*)&W, 4);
    f.write((char*)&H, 4);
    f.write((char*)probs.data(), (std::streamsize)(probs.size() * sizeof(float)));
    f.close();
}

// ─── Grid loading (flat) ──────────────────────────────────────────────────

// Load initial terrain grid as flat H*W array of terrain codes.
// Returns empty vector if not found.
std::vector<int> load_grid_flat(const std::string& path, int want_round, int want_seed,
                                int expect_W, int expect_H)
{
    std::ifstream f(path, std::ios::binary);
    if (!f) return {};

    char magic[4]; f.read(magic, 4);
    uint16_t ver; f.read((char*)&ver, 2);
    uint32_t count; f.read((char*)&count, 4);

    for (uint32_t i = 0; i < count; i++)
    {
        int32_t round, seed, W, H;
        f.read((char*)&round, 4);
        f.read((char*)&seed, 4);
        f.read((char*)&W, 4);
        f.read((char*)&H, 4);

        if (round == want_round && seed == want_seed && W == expect_W && H == expect_H)
        {
            std::vector<int> grid(H * W);
            for (int j = 0; j < H * W; j++)
            {
                int32_t val;
                f.read((char*)&val, 4);
                grid[j] = val;
            }
            return grid;
        }
        else
        {
            f.seekg((int64_t)W * H * 4, std::ios::cur);
        }
    }
    return {};
}

// ─── Scoring ─────────────────────────────────────────────────────────────────

static constexpr float PROB_FLOOR = 0.005f;

inline float prob_at(const ProbGrid& g, int W, int y, int x, int c)
{
    return g[(y * W + x) * NUM_CLASSES + c];
}

// Apply floor and renormalize a single cell in-place.
// If terrain >= 0, use smart flooring (only floor reachable classes).
void apply_floor(float* probs, int terrain = -1)
{
    if (terrain >= 0)
    {
        for (int c = 0; c < NUM_CLASSES; c++)
        {
            if (class_reachable(terrain, c))
            {
                if (probs[c] < PROB_FLOOR) probs[c] = PROB_FLOOR;
            }
            else
            {
                probs[c] = 0.0f;
            }
        }
    }
    else
    {
        for (int c = 0; c < NUM_CLASSES; c++)
            if (probs[c] < PROB_FLOOR) probs[c] = PROB_FLOOR;
    }

    float total = 0;
    for (int c = 0; c < NUM_CLASSES; c++) total += probs[c];
    if (total > 0)
        for (int c = 0; c < NUM_CLASSES; c++) probs[c] /= total;
}

struct ScoreResult
{
    double weighted_kl;
    double score;
    int dynamic_cells;
};

// Compute weighted KL and score for a blended prediction against ground truth
ScoreResult compute_score(const ProbGrid& gt, const ProbGrid& pred, int W, int H)
{
    double total_entropy = 0, total_weighted_kl = 0;
    int dynamic_cells = 0;

    for (int y = 0; y < H; y++)
    {
        for (int x = 0; x < W; x++)
        {
            double ent = 0;
            for (int c = 0; c < NUM_CLASSES; c++)
            {
                float p = prob_at(gt, W, y, x, c);
                if (p > 0) ent -= p * log(p);
            }

            if (ent < 1e-6) continue;
            dynamic_cells++;
            total_entropy += ent;

            double kl = 0;
            for (int c = 0; c < NUM_CLASSES; c++)
            {
                float p = prob_at(gt, W, y, x, c);
                float q = prob_at(pred, W, y, x, c);
                if (p > 0) kl += p * log(p / std::max(q, 1e-10f));
            }
            total_weighted_kl += ent * kl;
        }
    }

    double wkl = total_entropy > 0 ? total_weighted_kl / total_entropy : 0;
    double s = std::max(0.0, std::min(100.0, 100.0 * exp(-3.0 * wkl)));
    return {wkl, s, dynamic_cells};
}

// ─── Blending ────────────────────────────────────────────────────────────────

// Blend predictions with per-class weights
// weights[c][m] = weight for class c, model m
// terrain: flat H*W terrain codes (empty = use naive flooring)
ProbGrid blend_predictions(const PredSet& preds, int W, int H,
                           const float weights[NUM_CLASSES][NUM_MODELS],
                           const bool available[NUM_MODELS],
                           const std::vector<int>& terrain = {})
{
    int n = H * W * NUM_CLASSES;
    ProbGrid out(n);
    bool has_terrain = (int)terrain.size() == H * W;

    for (int y = 0; y < H; y++)
    {
        for (int x = 0; x < W; x++)
        {
            float cell[NUM_CLASSES];
            for (int c = 0; c < NUM_CLASSES; c++)
            {
                // Compute effective weights for available models
                float w_sum = 0;
                for (int m = 0; m < NUM_MODELS; m++)
                    if (available[m] && preds.has_model[m])
                        w_sum += weights[c][m];

                float val = 0;
                if (w_sum > 0)
                {
                    for (int m = 0; m < NUM_MODELS; m++)
                    {
                        if (!available[m] || !preds.has_model[m]) continue;
                        float w = weights[c][m] / w_sum;
                        val += w * prob_at(preds.model[m], W, y, x, c);
                    }
                }
                else
                {
                    // Fallback: uniform over available models
                    int nav = 0;
                    for (int m = 0; m < NUM_MODELS; m++)
                        if (preds.has_model[m]) nav++;
                    if (nav > 0)
                    {
                        for (int m = 0; m < NUM_MODELS; m++)
                            if (preds.has_model[m])
                                val += prob_at(preds.model[m], W, y, x, c) / nav;
                    }
                    else
                    {
                        val = 1.0f / NUM_CLASSES;
                    }
                }
                cell[c] = val;
            }

            int t = has_terrain ? terrain[y * W + x] : -1;
            apply_floor(cell, t);

            for (int c = 0; c < NUM_CLASSES; c++)
                out[(y * W + x) * NUM_CLASSES + c] = cell[c];
        }
    }

    return out;
}

// ─── Data entry used for optimization ─────────────────────────────────────────

struct DataEntry
{
    int round, seed, W, H;
    const ProbGrid* gt;
    PredSet preds;
    std::vector<int> terrain;  // H*W initial terrain codes (empty if grids not loaded)
};

// ─── Per-class KL optimization helper ────────────────────────────────────────

// Evaluate the per-class weighted KL contribution for a given weight vector
// on a subset of data entries (those not in exclude_round, or all if exclude_round < 0).
// Returns the negative total weighted KL (higher is better).
double eval_class_weights(int cls,
                          const std::array<float, NUM_MODELS>& w,
                          const std::vector<DataEntry>& data,
                          int exclude_round)
{
    double total_weighted_kl = 0;

    for (auto& de : data)
    {
        if (de.round == exclude_round) continue;
        bool has_terrain = (int)de.terrain.size() == de.H * de.W;

        for (int y = 0; y < de.H; y++)
        {
            for (int x = 0; x < de.W; x++)
            {
                double ent = 0;
                for (int c2 = 0; c2 < NUM_CLASSES; c2++)
                {
                    float p = prob_at(*de.gt, de.W, y, x, c2);
                    if (p > 0) ent -= p * log(p);
                }
                if (ent < 1e-6) continue;

                float gt_c = prob_at(*de.gt, de.W, y, x, cls);
                if (gt_c <= 0) continue;

                // Smart reachability: skip unreachable class for this terrain
                if (has_terrain)
                {
                    int t = de.terrain[y * de.W + x];
                    if (!class_reachable(t, cls)) continue;
                }

                float w_sum = 0;
                float blended = 0;
                for (int m = 0; m < NUM_MODELS; m++)
                {
                    if (!de.preds.has_model[m]) continue;
                    w_sum += w[m];
                }
                if (w_sum > 0)
                {
                    for (int m = 0; m < NUM_MODELS; m++)
                    {
                        if (!de.preds.has_model[m]) continue;
                        blended += (w[m] / w_sum) * prob_at(de.preds.model[m], de.W, y, x, cls);
                    }
                }
                else
                {
                    blended = 1.0f / NUM_CLASSES;
                }
                blended = std::max(blended, PROB_FLOOR);

                total_weighted_kl += ent * gt_c * log(gt_c / (double)blended);
            }
        }
    }

    return -total_weighted_kl;
}

// ─── Main ────────────────────────────────────────────────────────────────────

int main(int argc, char* argv[])
{
    if (argc < 2)
    {
        printf("Usage: blend <ground_truth.bin> [options]\n");
        printf("  --data-dir <path>    Directory containing prediction files (default: data)\n");
        printf("  --predict <round>    Generate blended predictions for this round\n");
        printf("  --output <path>      Output path pattern (default: {data_dir}/pred_blend_r{R}_s{S}.bin)\n");
        printf("  --step <N>           Grid search denominator (default: 10, i.e. step 0.1)\n");
        printf("  --grids <path>       Path to grids.bin for terrain-aware flooring\n");
        return 1;
    }

    std::string gt_path = argv[1];
    std::string data_dir = "data";
    int predict_round = -1;
    std::string output_pattern;
    int grid_steps = 10;
    std::string grids_path;

    for (int i = 2; i < argc; i++)
    {
        if (!strcmp(argv[i], "--data-dir") && i + 1 < argc)
            data_dir = argv[++i];
        else if (!strcmp(argv[i], "--predict") && i + 1 < argc)
            predict_round = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--output") && i + 1 < argc)
            output_pattern = argv[++i];
        else if (!strcmp(argv[i], "--step") && i + 1 < argc)
            grid_steps = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--grids") && i + 1 < argc)
            grids_path = argv[++i];
    }

    // Ensure data_dir ends with /
    if (!data_dir.empty() && data_dir.back() != '/')
        data_dir += '/';

    printf("=== Ensemble Weight Optimizer ===\n");
    printf("  Ground truth: %s\n", gt_path.c_str());
    printf("  Data dir:     %s\n", data_dir.c_str());
    printf("  Grid steps:   %d (step size = %.2f)\n", grid_steps, 1.0 / grid_steps);
    if (!grids_path.empty())
        printf("  Grids:        %s (smart flooring enabled)\n", grids_path.c_str());
    else
        printf("  Grids:        none (naive flooring — use --grids for better scores)\n");
    if (predict_round >= 0)
        printf("  Predict:      round %d\n", predict_round);
    printf("\n");

    // ─── Load ground truth ───────────────────────────────────────────────────

    auto gt_entries = load_all_ground_truth(gt_path);
    if (gt_entries.empty())
    {
        printf("Error: no ground truth entries loaded\n");
        return 1;
    }
    printf("Loaded %d ground truth entries\n", (int)gt_entries.size());

    // Gather unique rounds
    std::set<int> round_set;
    for (auto& e : gt_entries) round_set.insert(e.round);
    std::vector<int> rounds(round_set.begin(), round_set.end());
    std::sort(rounds.begin(), rounds.end());

    printf("Rounds:");
    for (int r : rounds) printf(" %d", r);
    printf("\n\n");

    // ─── Load model predictions for each GT entry ────────────────────────────

    std::vector<DataEntry> data;
    int model_counts[NUM_MODELS] = {};

    for (auto& gte : gt_entries)
    {
        DataEntry de;
        de.round = gte.round;
        de.seed = gte.seed;
        de.W = gte.W;
        de.H = gte.H;
        de.gt = &gte.probs;

        for (int m = 0; m < NUM_MODELS; m++)
        {
            char fname[256];
            snprintf(fname, sizeof(fname), "%spred_%s_r%d_s%d.bin",
                     data_dir.c_str(), MODEL_NAMES[m], gte.round, gte.seed);

            ProbGrid pg;
            if (load_astp(fname, gte.W, gte.H, pg))
            {
                de.preds.model[m] = std::move(pg);
                de.preds.has_model[m] = true;
                model_counts[m]++;
            }
        }

        // Load terrain grid for smart flooring
        if (!grids_path.empty())
        {
            de.terrain = load_grid_flat(grids_path, gte.round, gte.seed, gte.W, gte.H);
        }

        data.push_back(std::move(de));
    }

    // Report terrain loading
    if (!grids_path.empty())
    {
        int loaded = 0;
        for (auto& de : data) if (!de.terrain.empty()) loaded++;
        printf("Terrain grids loaded: %d / %d entries\n", loaded, (int)data.size());
    }

    printf("Model availability:\n");
    for (int m = 0; m < NUM_MODELS; m++)
        printf("  %-8s: %d / %d entries\n", MODEL_NAMES[m], model_counts[m], (int)data.size());
    printf("\n");

    // ─── Evaluate individual model scores ────────────────────────────────────

    printf("Individual model scores:\n");
    printf("  %-8s", "Model");
    for (int r : rounds) printf("  R%-4d", r);
    printf("   Avg\n");

    for (int m = 0; m < NUM_MODELS; m++)
    {
        printf("  %-8s", MODEL_NAMES[m]);
        double total = 0;
        int count = 0;

        std::map<int, std::pair<double, int>> round_scores;
        for (auto& de : data)
        {
            if (!de.preds.has_model[m]) continue;
            auto res = compute_score(*de.gt, de.preds.model[m], de.W, de.H);
            round_scores[de.round].first += res.score;
            round_scores[de.round].second++;
            total += res.score;
            count++;
        }

        for (int r : rounds)
        {
            auto it = round_scores.find(r);
            if (it != round_scores.end() && it->second.second > 0)
                printf("  %5.2f", it->second.first / it->second.second);
            else
                printf("    N/A");
        }
        if (count > 0)
            printf("  %5.2f", total / count);
        else
            printf("    N/A");
        printf("\n");
    }
    printf("\n");

    // ─── Pre-enumerate simplex points ────────────────────────────────────────

    // Determine which models are actually available for optimization
    bool global_available[NUM_MODELS] = {};
    for (int m = 0; m < NUM_MODELS; m++)
        global_available[m] = (model_counts[m] > 0);

    int num_available = 0;
    int available_idx[NUM_MODELS];
    for (int m = 0; m < NUM_MODELS; m++)
        if (global_available[m])
            available_idx[num_available++] = m;

    printf("Optimizing over %d available models:", num_available);
    for (int i = 0; i < num_available; i++)
        printf(" %s", MODEL_NAMES[available_idx[i]]);
    printf("\n");

    if (num_available < 2)
    {
        printf("Need at least 2 models for blending. Exiting.\n");
        return 1;
    }

    // Generate simplex points for the available models
    std::vector<std::array<float, NUM_MODELS>> simplex_pts;
    {
        std::array<float, NUM_MODELS> current = {};
        std::function<void(int, int)> recurse = [&](int idx, int remaining)
        {
            if (idx == num_available - 1)
            {
                current[idx] = (float)remaining / grid_steps;
                // Map to full model space
                std::array<float, NUM_MODELS> full = {};
                for (int i = 0; i < num_available; i++)
                    full[available_idx[i]] = current[i];
                simplex_pts.push_back(full);
                return;
            }
            for (int k = 0; k <= remaining; k++)
            {
                current[idx] = (float)k / grid_steps;
                recurse(idx + 1, remaining - k);
            }
        };
        recurse(0, grid_steps);
    }

    printf("Simplex points: %d (per class)\n\n", (int)simplex_pts.size());

    // ─── Leave-one-round-out cross-validation ────────────────────────────────

    printf("Running leave-one-round-out cross-validation...\n\n");

    double fold_scores[32] = {};  // up to 32 folds

    for (int fi = 0; fi < (int)rounds.size(); fi++)
    {
        int test_round = rounds[fi];

        // For each class, find best weights on training data (all rounds except test_round)
        float fw[NUM_CLASSES][NUM_MODELS] = {};

        for (int cls = 0; cls < NUM_CLASSES; cls++)
        {
            double best_neg_kl = -1e30;
            std::array<float, NUM_MODELS> best_w = {};

            for (auto& w : simplex_pts)
            {
                double neg_kl = eval_class_weights(cls, w, data, test_round);
                if (neg_kl > best_neg_kl)
                {
                    best_neg_kl = neg_kl;
                    best_w = w;
                }
            }

            for (int m = 0; m < NUM_MODELS; m++)
                fw[cls][m] = best_w[m];
        }

        // Evaluate on held-out round
        double fold_total_score = 0;
        int fold_count = 0;

        for (auto& de : data)
        {
            if (de.round != test_round) continue;

            ProbGrid blended = blend_predictions(de.preds, de.W, de.H, fw, global_available, de.terrain);
            auto res = compute_score(*de.gt, blended, de.W, de.H);
            fold_total_score += res.score;
            fold_count++;
        }

        fold_scores[fi] = fold_count > 0 ? fold_total_score / fold_count : 0;
    }

    // Print fold results
    printf("  Fold (held-out)  Score\n");
    double cv_avg = 0;
    for (int fi = 0; fi < (int)rounds.size(); fi++)
    {
        printf("  Round %-3d        %5.2f\n", rounds[fi], fold_scores[fi]);
        cv_avg += fold_scores[fi];
    }
    cv_avg /= rounds.size();
    printf("  ─────────────────────\n");
    printf("  CV Average       %5.2f\n\n", cv_avg);

    // ─── Final weights: train on ALL data ────────────────────────────────────

    printf("Optimizing final weights on all data...\n");

    float final_weights[NUM_CLASSES][NUM_MODELS] = {};

    for (int cls = 0; cls < NUM_CLASSES; cls++)
    {
        double best_neg_kl = -1e30;
        std::array<float, NUM_MODELS> best_w = {};

        for (auto& w : simplex_pts)
        {
            double neg_kl = eval_class_weights(cls, w, data, -1);
            if (neg_kl > best_neg_kl)
            {
                best_neg_kl = neg_kl;
                best_w = w;
            }
        }

        for (int m = 0; m < NUM_MODELS; m++)
            final_weights[cls][m] = best_w[m];
    }

    // ─── Print weight table ──────────────────────────────────────────────────

    const char* class_names[NUM_CLASSES] = {
        "empty/pln", "settle", "port", "ruin", "forest", "mountain"
    };

    printf("\n");
    printf("Optimal per-class weights:\n");
    printf("  %-12s", "Class");
    for (int m = 0; m < NUM_MODELS; m++)
        if (global_available[m])
            printf("  %-8s", MODEL_NAMES[m]);
    printf("\n");
    printf("  ────────────");
    for (int m = 0; m < NUM_MODELS; m++)
        if (global_available[m])
            printf("  ────────");
    printf("\n");

    for (int c = 0; c < NUM_CLASSES; c++)
    {
        printf("  %-12s", class_names[c]);
        for (int m = 0; m < NUM_MODELS; m++)
            if (global_available[m])
                printf("  %6.2f  ", final_weights[c][m]);
        printf("\n");
    }
    printf("\n");

    // ─── Evaluate final blend on all data ────────────────────────────────────

    printf("Final blend scores (in-sample):\n");
    printf("  %-8s", "Entry");
    printf("  %-8s", "Blend");
    for (int m = 0; m < NUM_MODELS; m++)
        if (global_available[m])
            printf("  %-8s", MODEL_NAMES[m]);
    printf("\n");

    double total_blend = 0, total_models[NUM_MODELS] = {};
    int count_blend = 0, count_models[NUM_MODELS] = {};

    for (auto& de : data)
    {
        ProbGrid blended = blend_predictions(de.preds, de.W, de.H, final_weights, global_available, de.terrain);
        auto res = compute_score(*de.gt, blended, de.W, de.H);

        char label[32];
        snprintf(label, sizeof(label), "R%d.S%d", de.round, de.seed);
        printf("  %-8s  %5.2f  ", label, res.score);

        total_blend += res.score;
        count_blend++;

        for (int m = 0; m < NUM_MODELS; m++)
        {
            if (!global_available[m]) continue;
            if (de.preds.has_model[m])
            {
                auto mres = compute_score(*de.gt, de.preds.model[m], de.W, de.H);
                printf("  %5.2f  ", mres.score);
                total_models[m] += mres.score;
                count_models[m]++;
            }
            else
            {
                printf("    N/A  ");
            }
        }
        printf("\n");
    }

    printf("  ────────────────");
    for (int m = 0; m < NUM_MODELS; m++)
        if (global_available[m])
            printf("──────────");
    printf("\n");

    printf("  %-8s  %5.2f  ", "Average", count_blend > 0 ? total_blend / count_blend : 0.0);
    for (int m = 0; m < NUM_MODELS; m++)
    {
        if (!global_available[m]) continue;
        if (count_models[m] > 0)
            printf("  %5.2f  ", total_models[m] / count_models[m]);
        else
            printf("    N/A  ");
    }
    printf("\n\n");

    // Compute improvement
    double best_single = 0;
    const char* best_model_name = "none";
    for (int m = 0; m < NUM_MODELS; m++)
    {
        if (!global_available[m] || count_models[m] == 0) continue;
        double avg = total_models[m] / count_models[m];
        if (avg > best_single)
        {
            best_single = avg;
            best_model_name = MODEL_NAMES[m];
        }
    }
    double blend_avg = count_blend > 0 ? total_blend / count_blend : 0;
    printf("Improvement over best single model (%s): %.2f -> %.2f (%+.2f)\n",
           best_model_name, best_single, blend_avg, blend_avg - best_single);
    printf("CV estimate: %.2f\n\n", cv_avg);

    // ─── Generate predictions for target round ───────────────────────────────

    if (predict_round >= 0)
    {
        printf("Generating blended predictions for round %d...\n", predict_round);

        // Find all seeds for this round from GT and by probing prediction files
        std::set<int> seeds_found;

        for (auto& gte : gt_entries)
            if (gte.round == predict_round)
                seeds_found.insert(gte.seed);

        // Also probe prediction files for seeds 0..9
        for (int s = 0; s < 10; s++)
        {
            for (int m = 0; m < NUM_MODELS; m++)
            {
                char fname[256];
                snprintf(fname, sizeof(fname), "%spred_%s_r%d_s%d.bin",
                         data_dir.c_str(), MODEL_NAMES[m], predict_round, s);
                std::ifstream test(fname);
                if (test.good()) { seeds_found.insert(s); break; }
            }
        }

        for (int seed : seeds_found)
        {
            int W = 40, H = 40;  // default grid size

            PredSet preds;
            for (int m = 0; m < NUM_MODELS; m++)
            {
                char fname[256];
                snprintf(fname, sizeof(fname), "%spred_%s_r%d_s%d.bin",
                         data_dir.c_str(), MODEL_NAMES[m], predict_round, seed);

                ProbGrid pg;
                if (load_astp(fname, W, H, pg))
                {
                    preds.model[m] = std::move(pg);
                    preds.has_model[m] = true;
                }
            }

            // Check we have at least one model
            int n_avail = 0;
            for (int m = 0; m < NUM_MODELS; m++)
                if (preds.has_model[m]) n_avail++;

            if (n_avail == 0)
            {
                printf("  R%d.S%d: no model predictions found, skipping\n", predict_round, seed);
                continue;
            }

            // Load terrain for smart flooring in prediction
            std::vector<int> pred_terrain;
            if (!grids_path.empty())
                pred_terrain = load_grid_flat(grids_path, predict_round, seed, W, H);

            ProbGrid blended = blend_predictions(preds, W, H, final_weights, global_available, pred_terrain);

            // Determine output path
            std::string out_path;
            if (!output_pattern.empty())
            {
                out_path = output_pattern;
                auto pos = out_path.find("{R}");
                if (pos != std::string::npos)
                    out_path.replace(pos, 3, std::to_string(predict_round));
                pos = out_path.find("{S}");
                if (pos != std::string::npos)
                    out_path.replace(pos, 3, std::to_string(seed));
            }
            else
            {
                char buf[256];
                snprintf(buf, sizeof(buf), "%spred_blend_r%d_s%d.bin",
                         data_dir.c_str(), predict_round, seed);
                out_path = buf;
            }

            write_astp(out_path, predict_round, seed, W, H, blended);

            // If we have GT, report score
            bool has_gt = false;
            for (auto& gte : gt_entries)
            {
                if (gte.round == predict_round && gte.seed == seed)
                {
                    auto res = compute_score(gte.probs, blended, W, H);
                    printf("  R%d.S%d: score=%.2f (wKL=%.6f) -> %s\n",
                           predict_round, seed, res.score, res.weighted_kl, out_path.c_str());
                    has_gt = true;
                    break;
                }
            }
            if (!has_gt)
                printf("  R%d.S%d: written to %s (%d models)\n",
                       predict_round, seed, out_path.c_str(), n_avail);
        }
        printf("\n");
    }

    printf("Done.\n");
    return 0;
}
