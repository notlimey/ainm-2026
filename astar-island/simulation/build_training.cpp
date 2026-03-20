#include <iostream>
#include <fstream>
#include <string>
#include <vector>
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

static const char MAGIC[] = "ASTF";
static const uint16_t VERSION = 1;

void write_training_data(const std::string& path, const std::vector<TrainingSample>& samples)
{
    std::ofstream f(path, std::ios::binary);
    if(!f) { printf("Error: cannot write %s\n", path.c_str()); return; }

    f.write(MAGIC, 4);
    uint16_t ver = VERSION;
    uint32_t n = samples.size();
    uint16_t feat_size = sizeof(CellFeatures);
    uint16_t nc = NUM_CLASSES;
    f.write((char*)&ver, 2);
    f.write((char*)&n, 4);
    f.write((char*)&feat_size, 2);
    f.write((char*)&nc, 2);

    for(auto& s : samples)
    {
        f.write((char*)&s.features, sizeof(CellFeatures));
        f.write((char*)s.ground_truth, sizeof(float) * NUM_CLASSES);
    }

    f.close();
    printf("Wrote %u samples to %s (%zu bytes per sample)\n",
        n, path.c_str(), sizeof(CellFeatures) + sizeof(float) * NUM_CLASSES);
}

bool load_analysis(const fs::path& path,
                   std::vector<std::vector<int>>& initial_grid,
                   std::vector<std::vector<std::vector<float>>>& ground_truth,
                   int& W, int& H)
{
    std::ifstream f(path);
    if(!f) return false;

    json j = json::parse(f);
    W = j["width"];
    H = j["height"];

    if(j["initial_grid"].is_null()) return false;
    initial_grid.resize(H);
    for(int y = 0; y < H; y++)
    {
        initial_grid[y].resize(W);
        for(int x = 0; x < W; x++)
            initial_grid[y][x] = j["initial_grid"][y][x];
    }

    ground_truth.resize(H);
    for(int y = 0; y < H; y++)
    {
        ground_truth[y].resize(W);
        for(int x = 0; x < W; x++)
        {
            ground_truth[y][x].resize(NUM_CLASSES);
            for(int c = 0; c < NUM_CLASSES; c++)
                ground_truth[y][x][c] = j["ground_truth"][y][x][c].get<float>();
        }
    }

    return true;
}

int main(int argc, char* argv[])
{
    fs::path analysis_dir = "../aggregate-data/data/analysis";
    fs::path out_path     = "data/training.bin";

    if(argc >= 2) analysis_dir = argv[1];
    if(argc >= 3) out_path     = argv[2];

    printf("Analysis dir: %s\n", analysis_dir.string().c_str());
    printf("Output:       %s\n\n", out_path.string().c_str());

    std::vector<TrainingSample> all_samples;
    int files_loaded = 0;

    std::vector<std::pair<int, fs::path>> round_dirs;
    for(auto& entry : fs::directory_iterator(analysis_dir))
    {
        if(!entry.is_directory()) continue;
        std::string name = entry.path().filename().string();
        if(name[0] != 'r') continue;
        try { round_dirs.push_back({std::stoi(name.substr(1)), entry.path()}); }
        catch(...) { continue; }
    }
    std::sort(round_dirs.begin(), round_dirs.end());

    for(auto& [round_num, round_path] : round_dirs)
    {
        for(auto& entry : fs::directory_iterator(round_path))
        {
            std::string name = entry.path().filename().string();
            if(name[0] != 's' || entry.path().extension() != ".json") continue;
            int seed = std::stoi(name.substr(1));

            std::vector<std::vector<int>> grid;
            std::vector<std::vector<std::vector<float>>> gt;
            int W, H;

            if(!load_analysis(entry.path(), grid, gt, W, H))
            {
                printf("  [r%d.s%d] skipped (no initial_grid)\n", round_num, seed);
                continue;
            }

            auto features = extract_features(grid);

            int dynamic_count = 0;
            for(int y = 0; y < H; y++)
            {
                for(int x = 0; x < W; x++)
                {
                    TrainingSample s;
                    s.features = features[y * W + x];
                    for(int c = 0; c < NUM_CLASSES; c++)
                        s.ground_truth[c] = gt[y][x][c];

                    bool is_dynamic = false;
                    for(int c = 1; c < NUM_CLASSES; c++)
                        if(s.ground_truth[c] > 0.01f) is_dynamic = true;
                    if(s.features.is_land && s.ground_truth[0] < 0.95f)
                        is_dynamic = true;

                    if(is_dynamic) dynamic_count++;
                    all_samples.push_back(s);
                }
            }

            printf("  [r%d.s%d] %dx%d = %d cells (%d dynamic)\n",
                round_num, seed, W, H, W*H, dynamic_count);
            files_loaded++;
        }
    }

    printf("\n%d files loaded, %d total samples\n", files_loaded, (int)all_samples.size());

    int total_dynamic = 0, total_ocean = 0, total_mountain = 0;
    int total_forest = 0, total_settlement = 0;
    for(auto& s : all_samples)
    {
        if(s.features.terrain_class == 0 && !s.features.is_land) total_ocean++;
        if(s.features.terrain_class == 5) total_mountain++;
        if(s.features.terrain_class == 4) total_forest++;
        if(s.features.terrain_class == 1 || s.features.terrain_class == 2) total_settlement++;

        float max_p = 0;
        for(int c = 0; c < NUM_CLASSES; c++)
            if(s.ground_truth[c] > max_p) max_p = s.ground_truth[c];
        if(max_p < 0.95f) total_dynamic++;
    }

    printf("\nDataset breakdown:\n");
    printf("  Ocean:       %d\n", total_ocean);
    printf("  Mountain:    %d\n", total_mountain);
    printf("  Forest:      %d\n", total_forest);
    printf("  Settlement:  %d\n", total_settlement);
    printf("  Dynamic:     %d (max_prob < 0.95)\n", total_dynamic);
    fs::create_directories(fs::path(out_path).parent_path());
    write_training_data(out_path.string(), all_samples);

    return 0;
}
