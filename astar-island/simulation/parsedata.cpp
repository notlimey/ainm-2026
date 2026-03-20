#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <nlohmann/json.hpp>
#include "database.hpp"
#include <filesystem>

namespace fs = std::filesystem;
using json = nlohmann::json;

Settlement parse_settlement(const json& j)
{
    Settlement s;
    s.x          = j["x"];
    s.y          = j["y"];
    s.has_port   = j["has_port"];
    s.alive      = j["alive"];
    s.population = j.value("population", 0.0f);
    s.food       = j.value("food",       0.0f);
    s.wealth     = j.value("wealth",     0.0f);
    s.defense    = j.value("defense",    0.0f);
    s.owner_id   = j.value("owner_id",  -1);
    return s;
}

WorldState parse_step(const fs::path& path, int round, int seed, int step)
{
    std::ifstream f(path);
    if(!f)
    {
        printf("  Skipping %s — not found\n", path.string().c_str());
        return WorldState();
    }

    json j = json::parse(f);
    WorldState state;
    state.round = round;
    state.seed  = seed;
    state.year  = step;

    for(auto& row : j["grid"])
    {
        std::vector<int> r;
        for(auto& cell : row) r.push_back(cell.get<int>());
        state.grid.push_back(r);
    }

    state.height = state.grid.size();
    state.width  = state.height > 0 ? state.grid[0].size() : 0;

    for(auto& s : j["settlements"])
        state.settlements.push_back(parse_settlement(s));

    return state;
}

int main(int argc, char* argv[])
{
    // Default: read from aggregate-data/data/rounds/{round}/{seed}/step_{N}.json
    fs::path base_dir = fs::path(argv[0]).parent_path().parent_path()
                        / "aggregate-data" / "data" / "rounds";
    fs::path out_dir  = fs::path(argv[0]).parent_path() / "data";

    if(argc >= 2) base_dir = argv[1];
    if(argc >= 3) out_dir  = argv[2];

    fs::create_directories(out_dir);

    printf("Input:  %s\n", base_dir.string().c_str());
    printf("Output: %s\n", out_dir.string().c_str());

    if(!fs::exists(base_dir))
    {
        printf("Error: input directory does not exist\n");
        return 1;
    }

    // Discover round directories (e.g. 1, 2, 3...)
    std::vector<std::pair<int, fs::path>> round_dirs;
    for(auto& entry : fs::directory_iterator(base_dir))
    {
        if(!entry.is_directory()) continue;
        try {
            int round_num = std::stoi(entry.path().filename().string());
            round_dirs.push_back({round_num, entry.path()});
        } catch(...) { continue; }
    }
    std::sort(round_dirs.begin(), round_dirs.end());

    std::vector<WorldState> all_states;
    int total_files = 0;

    for(auto& [round_num, round_path] : round_dirs)
    {
        // Discover seed directories (e.g. 0, 1, 2...)
        std::vector<std::pair<int, fs::path>> seed_dirs;
        for(auto& entry : fs::directory_iterator(round_path))
        {
            if(!entry.is_directory()) continue;
            try {
                int seed_idx = std::stoi(entry.path().filename().string());
                seed_dirs.push_back({seed_idx, entry.path()});
            } catch(...) { continue; }
        }
        std::sort(seed_dirs.begin(), seed_dirs.end());

        for(auto& [seed_idx, seed_path] : seed_dirs)
        {
            // Discover step files
            std::vector<std::pair<int, fs::path>> step_files;
            for(auto& entry : fs::directory_iterator(seed_path))
            {
                std::string name = entry.path().filename().string();
                if(name.substr(0, 5) != "step_" || entry.path().extension() != ".json")
                    continue;
                try {
                    int step = std::stoi(name.substr(5, name.size() - 10)); // strip "step_" and ".json"
                    step_files.push_back({step, entry.path()});
                } catch(...) { continue; }
            }
            std::sort(step_files.begin(), step_files.end());

            printf("Round %d, Seed %d: %d steps\n",
                round_num, seed_idx, (int)step_files.size());

            for(auto& [step, step_path] : step_files)
            {
                WorldState state = parse_step(step_path, round_num, seed_idx, step);
                if(!state.grid.empty())
                {
                    all_states.push_back(std::move(state));
                    total_files++;
                }
            }
        }
    }

    printf("\nParsed %d new step files\n", total_files);

    fs::path db_path = out_dir / "database.bin";

    // Merge with existing database if it exists
    Database db(db_path.string());
    std::vector<WorldState> merged;

    if(fs::exists(db_path))
    {
        db.load_index();
        merged = db.get_all();
        printf("Loaded %d existing records from %s\n",
            (int)merged.size(), db_path.string().c_str());
    }

    // Add new states, skip duplicates (same round/seed/step)
    int added = 0, skipped = 0;
    for(auto& state : all_states)
    {
        bool duplicate = false;
        for(auto& existing : merged)
        {
            if(existing.round == state.round &&
               existing.seed  == state.seed  &&
               existing.year  == state.year)
            {
                duplicate = true;
                break;
            }
        }
        if(!duplicate)
        {
            merged.push_back(std::move(state));
            added++;
        }
        else skipped++;
    }

    printf("Added %d new, skipped %d duplicates, total %d records\n",
        added, skipped, (int)merged.size());

    if(merged.empty())
    {
        printf("No data to write.\n");
        return 1;
    }

    printf("Building database -> %s\n", db_path.string().c_str());
    db.build(merged);
    db.load_index();
    db.print_summary();

    return 0;
}
