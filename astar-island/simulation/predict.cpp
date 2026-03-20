#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <map>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <nlohmann/json.hpp>
#include "features.hpp"

namespace fs = std::filesystem;
using json = nlohmann::json;

struct TrainingSample
{
    CellFeatures features;
    float ground_truth[NUM_CLASSES];
};

struct BucketKey
{
    uint8_t terrain_class;
    uint8_t is_coastal;
    uint8_t adj_settlement;
    uint8_t dist_settle_bin;
    uint8_t adj_forest;
    uint8_t adj_ocean;

    bool operator<(const BucketKey& o) const
    {
        if(terrain_class != o.terrain_class) return terrain_class < o.terrain_class;
        if(is_coastal != o.is_coastal) return is_coastal < o.is_coastal;
        if(adj_settlement != o.adj_settlement) return adj_settlement < o.adj_settlement;
        if(dist_settle_bin != o.dist_settle_bin) return dist_settle_bin < o.dist_settle_bin;
        if(adj_forest != o.adj_forest) return adj_forest < o.adj_forest;
        return adj_ocean < o.adj_ocean;
    }
};

BucketKey make_key(const CellFeatures& f)
{
    BucketKey k;
    k.terrain_class  = f.terrain_class;
    k.is_coastal     = f.is_coastal;
    k.adj_settlement = std::min(f.adj_settlement, (uint8_t)3);
    k.adj_forest     = std::min(f.adj_forest, (uint8_t)2);
    k.adj_ocean      = std::min(f.adj_ocean, (uint8_t)2);

    if(f.dist_nearest_settlement <= 1.5f)      k.dist_settle_bin = 0;
    else if(f.dist_nearest_settlement <= 3.5f)  k.dist_settle_bin = 1;
    else if(f.dist_nearest_settlement <= 6.5f)  k.dist_settle_bin = 2;
    else                                        k.dist_settle_bin = 3;

    return k;
}

BucketKey make_key_coarse(const CellFeatures& f)
{
    BucketKey k;
    k.terrain_class  = f.terrain_class;
    k.is_coastal     = f.is_coastal;
    k.adj_settlement = std::min(f.adj_settlement, (uint8_t)1);
    k.adj_forest     = 0;
    k.adj_ocean      = 0;

    if(f.dist_nearest_settlement <= 2.0f)       k.dist_settle_bin = 0;
    else if(f.dist_nearest_settlement <= 5.0f)   k.dist_settle_bin = 1;
    else                                         k.dist_settle_bin = 2;

    return k;
}

struct Bucket
{
    double sum[NUM_CLASSES] = {};
    int count = 0;

    void add(const float* gt)
    {
        for(int c = 0; c < NUM_CLASSES; c++)
            sum[c] += gt[c];
        count++;
    }

    void get_avg(float* out) const
    {
        if(count == 0)
        {
            for(int c = 0; c < NUM_CLASSES; c++)
                out[c] = 1.0f / NUM_CLASSES;
            return;
        }
        for(int c = 0; c < NUM_CLASSES; c++)
            out[c] = (float)(sum[c] / count);
    }
};

std::vector<TrainingSample> load_training(const std::string& path)
{
    std::ifstream f(path, std::ios::binary);
    if(!f) { printf("Error: cannot open %s\n", path.c_str()); return {}; }

    char magic[4];
    f.read(magic, 4);
    uint16_t ver; f.read((char*)&ver, 2);
    uint32_t n;   f.read((char*)&n, 4);
    uint16_t fs;  f.read((char*)&fs, 2);
    uint16_t nc;  f.read((char*)&nc, 2);

    printf("Loading %u training samples (v%d, %d features bytes, %d classes)\n", n, ver, fs, nc);

    std::vector<TrainingSample> samples(n);
    for(uint32_t i = 0; i < n; i++)
    {
        f.read((char*)&samples[i].features, sizeof(CellFeatures));
        f.read((char*)samples[i].ground_truth, sizeof(float) * NUM_CLASSES);
    }
    return samples;
}

void apply_floor(float* probs, float floor_val = 0.01f)
{
    for(int c = 0; c < NUM_CLASSES; c++)
        if(probs[c] < floor_val) probs[c] = floor_val;

    float total = 0;
    for(int c = 0; c < NUM_CLASSES; c++) total += probs[c];
    for(int c = 0; c < NUM_CLASSES; c++) probs[c] /= total;
}

int main(int argc, char* argv[])
{
    if(argc < 3)
    {
        printf("Usage: predict <training.bin> <initial_state.json> [output.json]\n");
        printf("\n");
        printf("  training.bin       - from build_training\n");
        printf("  initial_state.json - analysis JSON (uses initial_grid) or round detail\n");
        printf("  output.json        - prediction tensor [y][x][6] (default: stdout stats)\n");
        return 1;
    }

    std::string train_path  = argv[1];
    std::string input_path  = argv[2];
    std::string output_path = argc >= 4 ? argv[3] : "";

    auto samples = load_training(train_path);
    if(samples.empty()) return 1;

    std::map<BucketKey, Bucket> buckets;
    std::map<BucketKey, Bucket> buckets_coarse;

    for(auto& s : samples)
    {
        buckets[make_key(s.features)].add(s.ground_truth);
        buckets_coarse[make_key_coarse(s.features)].add(s.ground_truth);
    }

    printf("Built %d fine buckets, %d coarse buckets\n\n",
        (int)buckets.size(), (int)buckets_coarse.size());

    std::ifstream f(input_path);
    json j = json::parse(f);

    std::vector<std::vector<int>> grid;
    json& grid_json = j.contains("initial_grid") && !j["initial_grid"].is_null()
                      ? j["initial_grid"]
                      : j["grid"];

    int H = grid_json.size();
    int W = grid_json[0].size();
    grid.resize(H);
    for(int y = 0; y < H; y++)
    {
        grid[y].resize(W);
        for(int x = 0; x < W; x++)
            grid[y][x] = grid_json[y][x];
    }
    printf("Input grid: %dx%d\n", W, H);

    auto features = extract_features(grid);
    std::vector<std::vector<std::vector<float>>> prediction(H,
        std::vector<std::vector<float>>(W, std::vector<float>(NUM_CLASSES)));

    int fine_hits = 0, coarse_hits = 0, fallbacks = 0;

    for(int y = 0; y < H; y++)
    {
        for(int x = 0; x < W; x++)
        {
            const CellFeatures& cf = features[y * W + x];
            float* pred = prediction[y][x].data();

            auto it = buckets.find(make_key(cf));
            if(it != buckets.end() && it->second.count >= 3)
            {
                it->second.get_avg(pred);
                fine_hits++;
            }
            else
            {
                auto it2 = buckets_coarse.find(make_key_coarse(cf));
                if(it2 != buckets_coarse.end())
                {
                    it2->second.get_avg(pred);
                    coarse_hits++;
                }
                else
                {
                    for(int c = 0; c < NUM_CLASSES; c++) pred[c] = 0.01f;
                    pred[cf.terrain_class] = 0.94f;
                    fallbacks++;
                }
            }

            apply_floor(pred);
        }
    }

    printf("Predictions: %d fine, %d coarse, %d fallback\n",
        fine_hits, coarse_hits, fallbacks);

    if(j.contains("ground_truth") && !j["ground_truth"].is_null())
    {
        double total_entropy = 0, total_weighted_kl = 0;
        int dynamic_cells = 0;

        for(int y = 0; y < H; y++)
        {
            for(int x = 0; x < W; x++)
            {
                double ent = 0;
                for(int c = 0; c < NUM_CLASSES; c++)
                {
                    float p = j["ground_truth"][y][x][c].get<float>();
                    if(p > 0) ent -= p * log(p);
                }

                if(ent < 1e-6) continue;
                dynamic_cells++;
                total_entropy += ent;

                double kl = 0;
                for(int c = 0; c < NUM_CLASSES; c++)
                {
                    float p = j["ground_truth"][y][x][c].get<float>();
                    float q = prediction[y][x][c];
                    if(p > 0) kl += p * log(p / std::max(q, 1e-10f));
                }
                total_weighted_kl += ent * kl;
            }
        }

        double weighted_kl = total_entropy > 0 ? total_weighted_kl / total_entropy : 0;
        double score = std::max(0.0, std::min(100.0, 100.0 * exp(-3.0 * weighted_kl)));
        printf("\nValidation (vs ground truth in same file):\n");
        printf("  Dynamic cells: %d\n", dynamic_cells);
        printf("  Weighted KL:   %.6f\n", weighted_kl);
        printf("  Score:         %.2f / 100\n", score);
    }

    if(!output_path.empty())
    {
        json out = json::array();
        for(int y = 0; y < H; y++)
        {
            json row = json::array();
            for(int x = 0; x < W; x++)
            {
                json cell = json::array();
                for(int c = 0; c < NUM_CLASSES; c++)
                    cell.push_back(prediction[y][x][c]);
                row.push_back(cell);
            }
            out.push_back(row);
        }

        std::ofstream of(output_path);
        of << out.dump();
        of.close();
        printf("Prediction written to %s\n", output_path.c_str());
    }

    return 0;
}
