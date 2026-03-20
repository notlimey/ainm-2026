#include <iostream>
#include <string>
#include <map>
#include <set>
#include <algorithm>
#include "../simulation/database.hpp"

const char* terrain_name(int t)
{
    switch(t)
    {
        case TERRAIN_EMPTY:      return "empty";
        case TERRAIN_SETTLEMENT: return "settlement";
        case TERRAIN_PORT:       return "port";
        case TERRAIN_RUIN:       return "ruin";
        case TERRAIN_FOREST:     return "forest";
        case TERRAIN_MOUNTAIN:   return "mountain";
        case TERRAIN_OCEAN:      return "ocean";
        case TERRAIN_PLAINS:     return "plains";
        default:                 return "unknown";
    }
}

// Human-readable ID: [r1.s0.step5]
std::string state_id(int round, int seed, int step)
{
    return "[r" + std::to_string(round) +
           ".s" + std::to_string(seed) +
           ".step" + std::to_string(step) + "]";
}

struct TerrainCounts
{
    std::map<int, int> counts;
    void count_grid(const std::vector<std::vector<int>>& grid)
    {
        for(auto& row : grid)
            for(int cell : row)
                counts[cell]++;
    }
};

void print_overview(Database& db)
{
    printf("========================================\n");
    printf("  DATABASE OVERVIEW\n");
    printf("========================================\n\n");

    auto rounds = db.get_rounds();
    printf("Rounds: %d\n", (int)rounds.size());

    for(int r : rounds)
    {
        auto seeds = db.get_seeds(r);
        printf("\n  Round %d — %d seed(s)\n", r, (int)seeds.size());

        for(int s : seeds)
        {
            auto steps = db.get_years(r, s);
            std::sort(steps.begin(), steps.end());
            printf("    Seed %d — %d steps [%d..%d]\n",
                s, (int)steps.size(),
                steps.front(), steps.back());
        }
    }
    printf("\n");
}

void print_step_table(Database& db, int round, int seed)
{
    auto steps = db.get_years(round, seed);
    std::sort(steps.begin(), steps.end());

    printf("========================================\n");
    printf("  STEP TABLE — Round %d, Seed %d\n", round, seed);
    printf("========================================\n\n");

    printf("%-20s %6s %6s %6s  %-s\n",
        "ID", "settle", "alive", "ports", "terrain breakdown");
    printf("%-20s %6s %6s %6s  %-s\n",
        "----", "------", "-----", "-----", "-----------------");

    for(int step : steps)
    {
        WorldState state = db.get(round, seed, step);
        if(state.grid.empty()) continue;

        int alive = 0, ports = 0;
        for(auto& s : state.settlements)
        {
            if(s.alive) alive++;
            if(s.has_port) ports++;
        }

        TerrainCounts tc;
        tc.count_grid(state.grid);

        // Compact terrain string: only non-zero, skip ocean
        std::string terrain_str;
        int terrain_order[] = {
            TERRAIN_PLAINS, TERRAIN_FOREST, TERRAIN_MOUNTAIN,
            TERRAIN_SETTLEMENT, TERRAIN_PORT, TERRAIN_RUIN, TERRAIN_OCEAN
        };
        for(int t : terrain_order)
        {
            if(tc.counts[t] > 0)
            {
                if(!terrain_str.empty()) terrain_str += " ";
                terrain_str += std::string(terrain_name(t)) + ":" + std::to_string(tc.counts[t]);
            }
        }

        std::string id = state_id(round, seed, step);
        printf("%-20s %6d %6d %6d  %s\n",
            id.c_str(),
            (int)state.settlements.size(), alive, ports,
            terrain_str.c_str());
    }
    printf("\n");
}

void print_diff(Database& db, int round, int seed, int step_a, int step_b)
{
    std::string id_a = state_id(round, seed, step_a);
    std::string id_b = state_id(round, seed, step_b);

    printf("========================================\n");
    printf("  DIFF: %s -> %s\n", id_a.c_str(), id_b.c_str());
    printf("========================================\n\n");

    WorldState a = db.get(round, seed, step_a);
    WorldState b = db.get(round, seed, step_b);

    if(a.grid.empty() || b.grid.empty())
    {
        printf("  Could not load one or both states\n\n");
        return;
    }

    // Grid changes
    int grid_changes = 0;
    std::map<std::string, int> transition_counts; // "plains->settlement": N
    for(int y = 0; y < a.height && y < b.height; y++)
    {
        for(int x = 0; x < a.width && x < b.width; x++)
        {
            if(a.grid[y][x] != b.grid[y][x])
            {
                grid_changes++;
                std::string key = std::string(terrain_name(a.grid[y][x]))
                                + " -> "
                                + terrain_name(b.grid[y][x]);
                transition_counts[key]++;
            }
        }
    }

    printf("Grid cells changed: %d / %d\n", grid_changes, a.width * a.height);
    if(grid_changes > 0)
    {
        printf("Transitions:\n");
        for(auto& [key, count] : transition_counts)
            printf("  %-30s x%d\n", key.c_str(), count);
    }

    // Settlement changes
    int settlements_added   = (int)b.settlements.size() - (int)a.settlements.size();
    int died = 0, born_port = 0;

    // Match settlements by position
    std::map<std::pair<int,int>, const Settlement*> a_map, b_map;
    for(auto& s : a.settlements) a_map[{s.x, s.y}] = &s;
    for(auto& s : b.settlements) b_map[{s.x, s.y}] = &s;

    int new_settlements = 0, removed_settlements = 0;
    for(auto& [pos, s] : b_map)
        if(a_map.find(pos) == a_map.end()) new_settlements++;
    for(auto& [pos, s] : a_map)
        if(b_map.find(pos) == b_map.end()) removed_settlements++;

    // Stat changes for matched settlements
    float pop_delta = 0, food_delta = 0, wealth_delta = 0;
    int matched = 0;
    for(auto& [pos, sb] : b_map)
    {
        auto it = a_map.find(pos);
        if(it != a_map.end())
        {
            const Settlement* sa = it->second;
            pop_delta    += sb->population - sa->population;
            food_delta   += sb->food       - sa->food;
            wealth_delta += sb->wealth     - sa->wealth;
            if(sa->alive && !sb->alive) died++;
            if(!sa->has_port && sb->has_port) born_port++;
            matched++;
        }
    }

    printf("\nSettlements: %d -> %d (new: %d, removed: %d)\n",
        (int)a.settlements.size(), (int)b.settlements.size(),
        new_settlements, removed_settlements);
    if(matched > 0)
    {
        printf("Across %d matched settlements:\n", matched);
        printf("  pop:    %+.2f total (%+.3f avg)\n", pop_delta, pop_delta / matched);
        printf("  food:   %+.2f total (%+.3f avg)\n", food_delta, food_delta / matched);
        printf("  wealth: %+.2f total (%+.3f avg)\n", wealth_delta, wealth_delta / matched);
        printf("  died: %d, gained port: %d\n", died, born_port);
    }
    printf("\n");
}

int main(int argc, char* argv[])
{
    std::string db_path = "simulation/data/database.bin";
    if(argc >= 2) db_path = argv[1];

    printf("Loading: %s\n\n", db_path.c_str());

    Database db(db_path);
    db.load_index();

    // 1. Overview
    print_overview(db);

    // 2. Step table for each round/seed
    auto rounds = db.get_rounds();
    for(int r : rounds)
    {
        auto seeds = db.get_seeds(r);
        for(int s : seeds)
            print_step_table(db, r, s);
    }

    // 3. Show a few diffs (step 0->1, 1->2, and 0->50 if available)
    if(!rounds.empty())
    {
        int r = rounds[0];
        auto seeds = db.get_seeds(r);
        if(!seeds.empty())
        {
            int s = seeds[0];
            auto steps = db.get_years(r, s);
            std::sort(steps.begin(), steps.end());

            if(steps.size() >= 2)
                print_diff(db, r, s, steps[0], steps[1]);
            if(steps.size() >= 3)
                print_diff(db, r, s, steps[1], steps[2]);
            if(steps.size() > 2)
                print_diff(db, r, s, steps.front(), steps.back());
        }
    }

    return 0;
}
