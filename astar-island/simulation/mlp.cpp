#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <map>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <random>
#include <algorithm>
#include "features.hpp"
#include "io.hpp"

// --- Network architecture ---
static const int NUM_INPUTS  = 23;  // 21 + x/W, y/H position features
static const int HIDDEN1     = 64;
static const int HIDDEN2     = 64;
static const int NUM_OUTPUTS = NUM_CLASSES; // 6

struct TrainingSample
{
    uint16_t round;
    uint16_t seed;
    CellFeatures features;
    float ground_truth[NUM_CLASSES];
};

// --- Feature normalization: CellFeatures → 21 floats ---

void features_to_input(const CellFeatures& cf, float* out,
                        int total_settlements, int total_ports,
                        int W = 40, int H = 40)
{
    int i = 0;
    // terrain one-hot (6)
    for(int c = 0; c < 6; c++)
        out[i++] = (cf.terrain_class == c) ? 1.0f : 0.0f;
    // binary flags (2)
    out[i++] = cf.is_land ? 1.0f : 0.0f;
    out[i++] = cf.is_coastal ? 1.0f : 0.0f;
    // adjacency counts / 8 (5)
    out[i++] = cf.adj_ocean      / 8.0f;
    out[i++] = cf.adj_forest     / 8.0f;
    out[i++] = cf.adj_mountain   / 8.0f;
    out[i++] = cf.adj_settlement / 8.0f;
    out[i++] = cf.adj_plains     / 8.0f;
    // distances / 40 clamped (3)
    out[i++] = fminf(cf.dist_nearest_settlement, 40.0f) / 40.0f;
    out[i++] = fminf(cf.dist_nearest_port, 40.0f)       / 40.0f;
    out[i++] = fminf(cf.dist_coast, 40.0f)              / 40.0f;
    // settlement density (2)
    out[i++] = cf.settlements_r3 / 10.0f;
    out[i++] = cf.settlements_r5 / 20.0f;
    // land density (1)
    out[i++] = cf.land_cells_5x5 / 25.0f;
    // global (2)
    out[i++] = total_settlements / 20.0f;
    out[i++] = total_ports       / 10.0f;
    // position features (2) — spatial signal the MLP was missing
    out[i++] = cf.x / (float)W;
    out[i++] = cf.y / (float)H;
}

// --- MLP ---

struct MLP
{
    float W1[HIDDEN1][NUM_INPUTS], b1[HIDDEN1];
    float W2[HIDDEN2][HIDDEN1],    b2[HIDDEN2];
    float W3[NUM_OUTPUTS][HIDDEN2],b3[NUM_OUTPUTS];

    // gradients
    float dW1[HIDDEN1][NUM_INPUTS], db1[HIDDEN1];
    float dW2[HIDDEN2][HIDDEN1],    db2[HIDDEN2];
    float dW3[NUM_OUTPUTS][HIDDEN2],db3[NUM_OUTPUTS];

    // Adam momentum (m) and velocity (v)
    float mW1[HIDDEN1][NUM_INPUTS], mb1[HIDDEN1];
    float mW2[HIDDEN2][HIDDEN1],    mb2[HIDDEN2];
    float mW3[NUM_OUTPUTS][HIDDEN2],mb3[NUM_OUTPUTS];
    float vW1[HIDDEN1][NUM_INPUTS], vb1[HIDDEN1];
    float vW2[HIDDEN2][HIDDEN1],    vb2[HIDDEN2];
    float vW3[NUM_OUTPUTS][HIDDEN2],vb3[NUM_OUTPUTS];
    int adam_t = 0; // timestep

    // activations (kept for backprop)
    float z1[HIDDEN1], h1[HIDDEN1];
    float z2[HIDDEN2], h2[HIDDEN2];
    float logits[NUM_OUTPUTS], out[NUM_OUTPUTS];

    void init(std::mt19937& rng)
    {
        auto fill = [&](float* w, int fan_in, int n, float scale)
        {
            std::normal_distribution<float> d(0.0f, scale);
            for(int i = 0; i < n; i++) w[i] = d(rng);
        };
        fill(&W1[0][0], NUM_INPUTS, HIDDEN1 * NUM_INPUTS, sqrtf(2.0f / NUM_INPUTS));
        fill(&W2[0][0], HIDDEN1,    HIDDEN2 * HIDDEN1,    sqrtf(2.0f / HIDDEN1));
        fill(&W3[0][0], HIDDEN2,    NUM_OUTPUTS * HIDDEN2, sqrtf(1.0f / HIDDEN2));
        memset(b1, 0, sizeof(b1));
        memset(b2, 0, sizeof(b2));
        memset(b3, 0, sizeof(b3));
        // zero Adam state
        memset(mW1, 0, sizeof(mW1)); memset(mb1, 0, sizeof(mb1));
        memset(mW2, 0, sizeof(mW2)); memset(mb2, 0, sizeof(mb2));
        memset(mW3, 0, sizeof(mW3)); memset(mb3, 0, sizeof(mb3));
        memset(vW1, 0, sizeof(vW1)); memset(vb1, 0, sizeof(vb1));
        memset(vW2, 0, sizeof(vW2)); memset(vb2, 0, sizeof(vb2));
        memset(vW3, 0, sizeof(vW3)); memset(vb3, 0, sizeof(vb3));
        adam_t = 0;
    }

    void forward(const float* input)
    {
        // layer 1: ReLU(W1*x + b1)
        for(int i = 0; i < HIDDEN1; i++)
        {
            float s = b1[i];
            for(int j = 0; j < NUM_INPUTS; j++) s += W1[i][j] * input[j];
            z1[i] = s;
            h1[i] = s > 0 ? s : 0;
        }
        // layer 2: ReLU(W2*h1 + b2)
        for(int i = 0; i < HIDDEN2; i++)
        {
            float s = b2[i];
            for(int j = 0; j < HIDDEN1; j++) s += W2[i][j] * h1[j];
            z2[i] = s;
            h2[i] = s > 0 ? s : 0;
        }
        // output: softmax(W3*h2 + b3)
        float maxl = -1e30f;
        for(int i = 0; i < NUM_OUTPUTS; i++)
        {
            float s = b3[i];
            for(int j = 0; j < HIDDEN2; j++) s += W3[i][j] * h2[j];
            logits[i] = s;
            if(s > maxl) maxl = s;
        }
        float esum = 0;
        for(int i = 0; i < NUM_OUTPUTS; i++)
        {
            out[i] = expf(logits[i] - maxl);
            esum += out[i];
        }
        for(int i = 0; i < NUM_OUTPUTS; i++) out[i] /= esum;
    }

    // returns KL divergence loss, accumulates gradients
    float backward(const float* input, const float* target)
    {
        // loss = sum_i target_i * log(target_i / out_i)
        // d(loss)/d(logit_i) = out_i - target_i
        float loss = 0;
        float dl[NUM_OUTPUTS];
        for(int i = 0; i < NUM_OUTPUTS; i++)
        {
            if(target[i] > 0)
                loss += target[i] * logf(target[i] / fmaxf(out[i], 1e-10f));
            dl[i] = out[i] - target[i];
        }

        // W3, b3 grads
        for(int i = 0; i < NUM_OUTPUTS; i++)
        {
            for(int j = 0; j < HIDDEN2; j++) dW3[i][j] += dl[i] * h2[j];
            db3[i] += dl[i];
        }

        // backprop through layer 2
        float dh2[HIDDEN2];
        for(int j = 0; j < HIDDEN2; j++)
        {
            float s = 0;
            for(int i = 0; i < NUM_OUTPUTS; i++) s += W3[i][j] * dl[i];
            dh2[j] = z2[j] > 0 ? s : 0;
        }
        for(int i = 0; i < HIDDEN2; i++)
        {
            for(int j = 0; j < HIDDEN1; j++) dW2[i][j] += dh2[i] * h1[j];
            db2[i] += dh2[i];
        }

        // backprop through layer 1
        float dh1[HIDDEN1];
        for(int j = 0; j < HIDDEN1; j++)
        {
            float s = 0;
            for(int i = 0; i < HIDDEN2; i++) s += W2[i][j] * dh2[i];
            dh1[j] = z1[j] > 0 ? s : 0;
        }
        for(int i = 0; i < HIDDEN1; i++)
        {
            for(int j = 0; j < NUM_INPUTS; j++) dW1[i][j] += dh1[i] * input[j];
            db1[i] += dh1[i];
        }

        return loss;
    }

    void zero_grad()
    {
        memset(dW1, 0, sizeof(dW1)); memset(db1, 0, sizeof(db1));
        memset(dW2, 0, sizeof(dW2)); memset(db2, 0, sizeof(db2));
        memset(dW3, 0, sizeof(dW3)); memset(db3, 0, sizeof(db3));
    }

    void update(float lr, int batch_sz,
                float beta1 = 0.9f, float beta2 = 0.999f, float eps = 1e-8f)
    {
        adam_t++;
        float scale = 1.0f / batch_sz;
        float bc1 = 1.0f - powf(beta1, adam_t); // bias correction
        float bc2 = 1.0f - powf(beta2, adam_t);

        auto adam_step = [&](float* w, float* g, float* m, float* v, int n)
        {
            for(int i = 0; i < n; i++)
            {
                float gi = g[i] * scale;
                m[i] = beta1 * m[i] + (1 - beta1) * gi;
                v[i] = beta2 * v[i] + (1 - beta2) * gi * gi;
                float mh = m[i] / bc1;
                float vh = v[i] / bc2;
                w[i] -= lr * mh / (sqrtf(vh) + eps);
            }
        };

        adam_step(&W1[0][0], &dW1[0][0], &mW1[0][0], &vW1[0][0], HIDDEN1 * NUM_INPUTS);
        adam_step(b1, db1, mb1, vb1, HIDDEN1);
        adam_step(&W2[0][0], &dW2[0][0], &mW2[0][0], &vW2[0][0], HIDDEN2 * HIDDEN1);
        adam_step(b2, db2, mb2, vb2, HIDDEN2);
        adam_step(&W3[0][0], &dW3[0][0], &mW3[0][0], &vW3[0][0], NUM_OUTPUTS * HIDDEN2);
        adam_step(b3, db3, mb3, vb3, NUM_OUTPUTS);
    }

    void save(const std::string& path)
    {
        std::ofstream f(path, std::ios::binary);
        f.write("MLPW", 4);
        int32_t dims[4] = {NUM_INPUTS, HIDDEN1, HIDDEN2, NUM_OUTPUTS};
        f.write((char*)dims, 16);
        f.write((char*)W1, sizeof(W1)); f.write((char*)b1, sizeof(b1));
        f.write((char*)W2, sizeof(W2)); f.write((char*)b2, sizeof(b2));
        f.write((char*)W3, sizeof(W3)); f.write((char*)b3, sizeof(b3));
        printf("Model saved: %s\n", path.c_str());
    }

    bool load(const std::string& path)
    {
        std::ifstream f(path, std::ios::binary);
        if(!f) return false;
        char magic[4]; f.read(magic, 4);
        if(memcmp(magic, "MLPW", 4)) { printf("Bad model magic\n"); return false; }
        int32_t dims[4]; f.read((char*)dims, 16);
        if(dims[0] != NUM_INPUTS || dims[1] != HIDDEN1 ||
           dims[2] != HIDDEN2 || dims[3] != NUM_OUTPUTS)
        { printf("Model architecture mismatch\n"); return false; }
        f.read((char*)W1, sizeof(W1)); f.read((char*)b1, sizeof(b1));
        f.read((char*)W2, sizeof(W2)); f.read((char*)b2, sizeof(b2));
        f.read((char*)W3, sizeof(W3)); f.read((char*)b3, sizeof(b3));
        printf("Model loaded: %s\n", path.c_str());
        return true;
    }
};

// --- I/O (same as predict.cpp) ---

std::vector<TrainingSample> load_training(const std::string& path)
{
    std::ifstream f(path, std::ios::binary);
    if(!f) { printf("Error: cannot open %s\n", path.c_str()); return {}; }
    char magic[4]; f.read(magic, 4);
    uint16_t ver; f.read((char*)&ver, 2);
    uint32_t n;   f.read((char*)&n, 4);
    uint16_t fs;  f.read((char*)&fs, 2);
    uint16_t nc;  f.read((char*)&nc, 2);
    printf("Training: %u samples (v%d)\n", n, ver);
    std::vector<TrainingSample> samples(n);
    for(uint32_t i = 0; i < n; i++)
    {
        if(ver >= 2) { f.read((char*)&samples[i].round, 2); f.read((char*)&samples[i].seed, 2); }
        else { samples[i].round = 0; samples[i].seed = 0; }
        f.read((char*)&samples[i].features, sizeof(CellFeatures));
        f.read((char*)samples[i].ground_truth, sizeof(float) * NUM_CLASSES);
    }
    return samples;
}

// I/O and apply_floor now in io.hpp

// --- Compute per-(round,seed) global stats from training samples ---

struct GridKey
{
    uint16_t round, seed;
    bool operator<(const GridKey& o) const
    {
        return round != o.round ? round < o.round : seed < o.seed;
    }
};

struct GridStats { int total_settlements = 0, total_ports = 0; };

std::map<GridKey, GridStats> compute_grid_stats(const std::vector<TrainingSample>& samples)
{
    std::map<GridKey, GridStats> stats;
    for(auto& s : samples)
    {
        GridKey k{s.round, s.seed};
        if(s.features.terrain_class == 1) stats[k].total_settlements++;
        if(s.features.terrain_class == 2) { stats[k].total_settlements++; stats[k].total_ports++; }
    }
    return stats;
}

int count_in_grid(const std::vector<std::vector<int>>& grid, int code)
{
    int n = 0;
    for(auto& row : grid) for(int v : row) if(v == code) n++;
    return n;
}

// --- Main ---

int main(int argc, char* argv[])
{
    if(argc < 5)
    {
        printf("Usage: mlp <training.bin> <grids.bin> <round> <seed> [prediction.bin]\n"
               "  [--exclude <round>] [--epochs N] [--lr F] [--batch N]\n"
               "  [--save-model path] [--load-model path]\n");
        return 1;
    }

    std::string train_path = argv[1];
    std::string grids_path = argv[2];
    int round = std::stoi(argv[3]);
    int seed  = std::stoi(argv[4]);

    std::string out_path, save_path, load_path;
    int exclude_round = -1;
    int epochs = 80;
    float lr = 0.001f;
    int batch_sz = 256;

    for(int i = 5; i < argc; i++)
    {
        std::string arg = argv[i];
        if(arg == "--exclude"    && i+1 < argc) exclude_round = std::stoi(argv[++i]);
        else if(arg == "--epochs"     && i+1 < argc) epochs = std::stoi(argv[++i]);
        else if(arg == "--lr"         && i+1 < argc) lr = std::stof(argv[++i]);
        else if(arg == "--batch"      && i+1 < argc) batch_sz = std::stoi(argv[++i]);
        else if(arg == "--save-model" && i+1 < argc) save_path = argv[++i];
        else if(arg == "--load-model" && i+1 < argc) load_path = argv[++i];
        else if(out_path.empty()) out_path = arg;
    }

    // --- Load training data ---
    auto samples = load_training(train_path);
    if(samples.empty()) return 1;

    MLP net;
    std::mt19937 rng(42);

    if(!load_path.empty() && net.load(load_path))
    {
        // skip training
    }
    else
    {
        // --- Prepare training inputs ---
        auto grid_stats = compute_grid_stats(samples);

        // Filter out excluded round, build input arrays
        std::vector<int> indices;
        indices.reserve(samples.size());
        int excluded = 0;
        for(int i = 0; i < (int)samples.size(); i++)
        {
            if(exclude_round >= 0 && samples[i].round == exclude_round)
                { excluded++; continue; }
            indices.push_back(i);
        }
        int N = indices.size();
        printf("Training MLP: %d samples", N);
        if(excluded > 0) printf(", %d excluded (round %d)", excluded, exclude_round);
        printf("\n  Architecture: %d → %d → %d → %d\n", NUM_INPUTS, HIDDEN1, HIDDEN2, NUM_OUTPUTS);
        printf("  Epochs: %d, Adam LR: %.4f, Batch: %d\n", epochs, lr, batch_sz);

        // Pre-compute all input vectors
        std::vector<std::vector<float>> inputs(samples.size(), std::vector<float>(NUM_INPUTS));
        for(int i : indices)
        {
            GridKey k{samples[i].round, samples[i].seed};
            auto it = grid_stats.find(k);
            int ts = it != grid_stats.end() ? it->second.total_settlements : 0;
            int tp = it != grid_stats.end() ? it->second.total_ports : 0;
            features_to_input(samples[i].features, inputs[i].data(), ts, tp, 40, 40);
        }

        // --- Train ---
        net.init(rng);

        for(int ep = 0; ep < epochs; ep++)
        {
            std::shuffle(indices.begin(), indices.end(), rng);

            double epoch_loss = 0;

            for(int b = 0; b < N; b += batch_sz)
            {
                int end = std::min(b + batch_sz, N);
                int bs = end - b;
                net.zero_grad();

                float batch_loss = 0;
                for(int k = b; k < end; k++)
                {
                    int idx = indices[k];
                    net.forward(inputs[idx].data());
                    batch_loss += net.backward(inputs[idx].data(), samples[idx].ground_truth);
                }

                net.update(lr, bs);
                epoch_loss += batch_loss;
            }

            if(ep % 10 == 0 || ep == epochs - 1)
                printf("  Epoch %3d/%d  loss=%.6f\n",
                       ep + 1, epochs, epoch_loss / N);
        }

        if(!save_path.empty()) net.save(save_path);
    }

    // --- Load grid and predict ---
    std::vector<std::vector<int>> grid;
    int W, H;
    if(!load_grid(grids_path, round, seed, grid, W, H)) return 1;
    printf("Grid [r%d.s%d]: %dx%d\n", round, seed, W, H);

    auto cell_features = extract_features(grid);
    int total_settle = count_in_grid(grid, 1) + count_in_grid(grid, 2);
    int total_port   = count_in_grid(grid, 2);

    std::vector<std::vector<std::vector<float>>> prediction(H,
        std::vector<std::vector<float>>(W, std::vector<float>(NUM_CLASSES)));

    float input_buf[NUM_INPUTS];
    for(int y = 0; y < H; y++)
    {
        for(int x = 0; x < W; x++)
        {
            features_to_input(cell_features[y * W + x], input_buf,
                              total_settle, total_port, W, H);
            net.forward(input_buf);

            float* pred = prediction[y][x].data();
            for(int c = 0; c < NUM_CLASSES; c++) pred[c] = net.out[c];
            apply_floor(pred);
        }
    }

    // --- Validate against ground truth ---
    std::vector<std::vector<std::vector<float>>> gt;
    int gtW, gtH;
    if(load_ground_truth("data/ground_truth.bin", round, seed, gt, gtW, gtH))
    {
        double total_entropy = 0, total_weighted_kl = 0;
        int dynamic_cells = 0;
        for(int y = 0; y < H && y < gtH; y++)
        {
            for(int x = 0; x < W && x < gtW; x++)
            {
                double ent = 0;
                for(int c = 0; c < NUM_CLASSES; c++)
                {
                    float p = gt[y][x][c];
                    if(p > 0) ent -= p * log(p);
                }
                if(ent < 1e-6) continue;
                dynamic_cells++;
                total_entropy += ent;

                double kl = 0;
                for(int c = 0; c < NUM_CLASSES; c++)
                {
                    float p = gt[y][x][c];
                    float q = prediction[y][x][c];
                    if(p > 0) kl += p * log(p / std::max(q, 1e-10f));
                }
                total_weighted_kl += ent * kl;
            }
        }
        double weighted_kl = total_entropy > 0 ? total_weighted_kl / total_entropy : 0;
        double score = std::max(0.0, std::min(100.0, 100.0 * exp(-3.0 * weighted_kl)));
        printf("\nValidation vs ground truth:\n");
        printf("  Dynamic cells: %d\n", dynamic_cells);
        printf("  Weighted KL:   %.6f\n", weighted_kl);
        printf("  Score:         %.2f / 100\n", score);
    }

    if(!out_path.empty())
    {
        write_prediction(out_path, round, seed, W, H, prediction);
        printf("Written to %s\n", out_path.c_str());
    }

    return 0;
}
