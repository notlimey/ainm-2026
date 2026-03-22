#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <map>
#include <cmath>
#include <cstdint>
#include "features.hpp"
#include "io.hpp"

struct TrainingSample
{
    uint16_t round;
    uint16_t seed;
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
    uint8_t settle_r3_bin;
    uint8_t settle_r5_bin;

    bool operator<(const BucketKey& o) const
    {
        if(terrain_class != o.terrain_class) return terrain_class < o.terrain_class;
        if(is_coastal != o.is_coastal) return is_coastal < o.is_coastal;
        if(adj_settlement != o.adj_settlement) return adj_settlement < o.adj_settlement;
        if(dist_settle_bin != o.dist_settle_bin) return dist_settle_bin < o.dist_settle_bin;
        if(adj_forest != o.adj_forest) return adj_forest < o.adj_forest;
        if(settle_r3_bin != o.settle_r3_bin) return settle_r3_bin < o.settle_r3_bin;
        return settle_r5_bin < o.settle_r5_bin;
    }
};

BucketKey make_key(const CellFeatures& f)
{
    BucketKey k;
    k.terrain_class  = f.terrain_class;
    k.is_coastal     = f.is_coastal;
    k.adj_settlement = std::min(f.adj_settlement, (uint8_t)3);
    k.adj_forest     = std::min(f.adj_forest, (uint8_t)2);
    k.settle_r3_bin  = std::min(f.settlements_r3, (uint8_t)4);
    k.settle_r5_bin  = std::min((uint8_t)(f.settlements_r5 / 2), (uint8_t)3);

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
    k.settle_r3_bin  = std::min((uint8_t)(f.settlements_r3 / 2), (uint8_t)2);
    k.settle_r5_bin  = 0;

    if(f.dist_nearest_settlement <= 1.5f)       k.dist_settle_bin = 0;
    else if(f.dist_nearest_settlement <= 3.5f)   k.dist_settle_bin = 1;
    else if(f.dist_nearest_settlement <= 6.5f)   k.dist_settle_bin = 2;
    else                                         k.dist_settle_bin = 3;

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

    printf("Training: %u samples (v%d)\n", n, ver);

    std::vector<TrainingSample> samples(n);
    for(uint32_t i = 0; i < n; i++)
    {
        if(ver >= 2)
        {
            f.read((char*)&samples[i].round, 2);
            f.read((char*)&samples[i].seed, 2);
        }
        else { samples[i].round = 0; samples[i].seed = 0; }
        f.read((char*)&samples[i].features, sizeof(CellFeatures));
        f.read((char*)samples[i].ground_truth, sizeof(float) * NUM_CLASSES);
    }
    return samples;
}

// I/O, scoring, and apply_floor now in io.hpp

int main(int argc, char* argv[])
{
    if(argc < 5)
    {
        printf("Usage: predict <training.bin> <grids.bin> <round> <seed> [prediction.bin] [--exclude <round>]\n");
        return 1;
    }

    std::string train_path = argv[1];
    std::string grids_path = argv[2];
    int round = std::stoi(argv[3]);
    int seed  = std::stoi(argv[4]);
    std::string out_path;
    int exclude_round = -1;

    for(int i = 5; i < argc; i++)
    {
        std::string arg = argv[i];
        if(arg == "--exclude" && i + 1 < argc)
            exclude_round = std::stoi(argv[++i]);
        else if(out_path.empty())
            out_path = arg;
    }

    auto samples = load_training(train_path);
    if(samples.empty()) return 1;

    std::map<BucketKey, Bucket> buckets;
    std::map<BucketKey, Bucket> buckets_coarse;

    int used = 0, excluded = 0;
    for(auto& s : samples)
    {
        if(exclude_round >= 0 && s.round == exclude_round) { excluded++; continue; }
        buckets[make_key(s.features)].add(s.ground_truth);
        buckets_coarse[make_key_coarse(s.features)].add(s.ground_truth);
        used++;
    }

    printf("Training: %d used", used);
    if(excluded > 0) printf(", %d excluded (round %d)", excluded, exclude_round);
    printf("\nBuckets: %d fine, %d coarse\n", (int)buckets.size(), (int)buckets_coarse.size());

    std::vector<std::vector<int>> grid;
    int W, H;
    if(!load_grid(grids_path, round, seed, grid, W, H)) return 1;
    printf("Grid [r%d.s%d]: %dx%d\n", round, seed, W, H);

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
            if(it != buckets.end() && it->second.count >= 10)
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

    std::string gt_path = "data/ground_truth.bin";
    std::vector<std::vector<std::vector<float>>> gt;
    int gtW, gtH;
    if(load_ground_truth(gt_path, round, seed, gt, gtW, gtH))
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
