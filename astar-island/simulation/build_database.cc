#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <nlohmann/json.hpp>
#include "database.hpp"
#include <filesystem>

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

WorldState parse_json(const std::string& path, int round, int seed, int year)
{
    std::ifstream f(path);
    if(!f)
    {
        printf("  Skipping %s — not found\n", path.c_str());
        return WorldState();
    }

    json j = json::parse(f);
    WorldState state;
    state.round = round;
    state.seed  = seed;
    state.year  = year;

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

    printf("  Parsed r=%d s=%d y=%d — grid=%dx%d settlements=%d\n",
        round, seed, year, state.width, state.height, 
        (int)state.settlements.size());
    return state;
}

int main(int argc, char* argv[])
{
    std::filesystem::path exe_path = std::filesystem::path(argv[0]).parent_path();
    std::filesystem::path data_dir = exe_path / "data" / "rounds";
    std::filesystem::path db_path  = exe_path / "data" / "database.bin";

    printf("Data directory: %s\n", data_dir.string().c_str());
    printf("Database path:  %s\n", db_path.string().c_str());

    std::vector<WorldState> states;

    printf("Loading JSON files...\n");
    for(int round = 1; round <= 5; round++)
    {
        for(int seed = 0; seed <= 4; seed++)
        {
            for(int year = 0; year <= 50; year++)
            {
                std::filesystem::path json_path = data_dir /
                    ("r" + std::to_string(round) +
                     "_s" + std::to_string(seed) +
                     "_y" + std::to_string(year) + ".json");

                WorldState state = parse_json(json_path.string(), round, seed, year);
                if(!state.grid.empty())
                    states.push_back(state);
            }
        }
    }

    printf("\nBuilding database with %d records...\n", (int)states.size());
    Database db(db_path.string());
    db.build(states);

    db.load_index();
    db.print_summary();

    printf("--- Testing lookups ---\n");
    printf("Cell at r=1 s=0 y=0 (0,0): %d\n", db.get_cell(1, 0, 0, 0, 0));
    printf("Cell at r=1 s=0 y=50 (20,20): %d\n", db.get_cell(1, 0, 50, 20, 20));

    auto years = db.get_years(1, 0);
    printf("\nYears available for r=1 s=0: ");
    for(int y : years) printf("%d ", y);
    printf("\n");

    auto ports = db.get_ports(1, 0, 0);
    printf("\nPorts at r=1 s=0 y=0: %d\n", (int)ports.size());
    for(auto& p : ports)
        printf("  (%d,%d)\n", p.x, p.y);

    auto alive = db.get_alive_settlements(1, 0, 50);
    printf("\nAlive settlements at r=1 s=0 y=50: %d\n", (int)alive.size());
    for(auto& s : alive)
        printf("  (%d,%d) pop=%.1f wealth=%.1f port=%s\n",
            s.x, s.y, s.population, s.wealth,
            s.has_port ? "yes" : "no");

    return 0;
}