#pragma once
#include <vector>
#include <string>
#include <fstream>
#include <map>

// ─── Terrain constants ───────────────────────────────────────────────────────
#define TERRAIN_EMPTY      0
#define TERRAIN_SETTLEMENT 1
#define TERRAIN_PORT       2
#define TERRAIN_RUIN       3
#define TERRAIN_FOREST     4
#define TERRAIN_MOUNTAIN   5
#define TERRAIN_OCEAN      10
#define TERRAIN_PLAINS     11

// ─── Data types ──────────────────────────────────────────────────────────────
struct Settlement
{
    int x, y;
    float population;
    float food;
    float wealth;
    float defense;
    bool has_port;
    bool alive;
    int owner_id;
};

struct WorldState
{
    int round;
    int seed;
    int year;
    int width;
    int height;
    std::vector<std::vector<int>> grid;
    std::vector<Settlement> settlements;
};


struct InitialState
{
    int round;
    int seed;
    int width;
    int height;
    std::vector<std::vector<int>> grid;
    std::vector<Settlement> settlements;
};

// ─── Index entry — tracks where each record is in the file ───────────────────
struct IndexEntry
{
    int round;
    int seed;
    int year;
    int64_t offset;
    int32_t size;
};

// ─── Database ────────────────────────────────────────────────────────────────
class Database
{
private:
    std::string path;
    std::vector<IndexEntry> index;

    // ─── Write helpers ───────────────────────────────────────────────────────
    static void write_int(std::ofstream& f, int val)
    {
        f.write((char*)&val, sizeof(int));
    }
    static void write_int64(std::ofstream& f, int64_t val)
    {
        f.write((char*)&val, sizeof(int64_t));
    }
    static void write_int32(std::ofstream& f, int32_t val)
    {
        f.write((char*)&val, sizeof(int32_t));
    }
    static void write_bool(std::ofstream& f, bool val)
    {
        f.write((char*)&val, sizeof(bool));
    }
    static void write_float(std::ofstream& f, float val)
    {
        f.write((char*)&val, sizeof(float));
    }
    static void write_grid(std::ofstream& f, const std::vector<std::vector<int>>& grid)
    {
        int h = grid.size();
        int w = h > 0 ? grid[0].size() : 0;
        write_int(f, h);
        write_int(f, w);
        for(int y = 0; y < h; y++)
            for(int x = 0; x < w; x++)
                write_int(f, grid[y][x]);
    }
    static void write_settlement(std::ofstream& f, const Settlement& s)
    {
        write_int(f,   s.x);
        write_int(f,   s.y);
        write_float(f, s.population);
        write_float(f, s.food);
        write_float(f, s.wealth);
        write_float(f, s.defense);
        write_bool(f,  s.has_port);
        write_bool(f,  s.alive);
        write_int(f,   s.owner_id);
    }

    // ─── Read helpers ────────────────────────────────────────────────────────
    static int read_int(std::ifstream& f)
    {
        int val; f.read((char*)&val, sizeof(int)); return val;
    }
    static int64_t read_int64(std::ifstream& f)
    {
        int64_t val; f.read((char*)&val, sizeof(int64_t)); return val;
    }
    static int32_t read_int32(std::ifstream& f)
    {
        int32_t val; f.read((char*)&val, sizeof(int32_t)); return val;
    }
    static bool read_bool(std::ifstream& f)
    {
        bool val; f.read((char*)&val, sizeof(bool)); return val;
    }
    static float read_float(std::ifstream& f)
    {
        float val; f.read((char*)&val, sizeof(float)); return val;
    }
    static std::vector<std::vector<int>> read_grid(std::ifstream& f)
    {
        int h = read_int(f);
        int w = read_int(f);
        std::vector<std::vector<int>> grid(h, std::vector<int>(w));
        for(int y = 0; y < h; y++)
            for(int x = 0; x < w; x++)
                grid[y][x] = read_int(f);
        return grid;
    }
    static Settlement read_settlement(std::ifstream& f)
    {
        Settlement s;
        s.x          = read_int(f);
        s.y          = read_int(f);
        s.population = read_float(f);
        s.food       = read_float(f);
        s.wealth     = read_float(f);
        s.defense    = read_float(f);
        s.has_port   = read_bool(f);
        s.alive      = read_bool(f);
        s.owner_id   = read_int(f);
        return s;
    }

    // ─── Index helpers ───────────────────────────────────────────────────────
    void write_index(std::ofstream& f)
    {
        write_int(f, index.size());
        for(auto& e : index)
        {
            write_int(f,   e.round);
            write_int(f,   e.seed);
            write_int(f,   e.year);
            write_int64(f, e.offset);
            write_int32(f, e.size);
        }
    }
    void read_index(std::ifstream& f)
    {
        index.clear();
        int count = read_int(f);
        for(int i = 0; i < count; i++)
        {
            IndexEntry e;
            e.round  = read_int(f);
            e.seed   = read_int(f);
            e.year   = read_int(f);
            e.offset = read_int64(f);
            e.size   = read_int32(f);
            index.push_back(e);
        }
    }
    IndexEntry* find_entry(int round, int seed, int year)
    {
        for(auto& e : index)
            if(e.round == round && e.seed == seed && e.year == year)
                return &e;
        return nullptr;
    }

    // ─── Record size calculator ───────────────────────────────────────────────
    static int32_t calc_record_size(const WorldState& state)
    {
        int32_t size = 0;
        size += sizeof(int) * 5;  // round, seed, year, width, height
        size += sizeof(int) * 2;  // grid h, w
        size += sizeof(int) * state.width * state.height;  // grid data
        size += sizeof(int);      // num settlements
        size += state.settlements.size() * (
            sizeof(int)   * 3 +   // x, y, owner_id
            sizeof(float) * 4 +   // population, food, wealth, defense
            sizeof(bool)  * 2     // has_port, alive
        );
        return size;
    }

public:
    Database(const std::string& path) : path(path) {}

    // ─── Build database from a list of WorldStates ───────────────────────────
    void build(const std::vector<WorldState>& states)
    {
        index.clear();

        // Calculate index block size
        int n = states.size();
        int64_t index_size = sizeof(int) + 
            n * (sizeof(int)*3 + sizeof(int64_t) + sizeof(int32_t));
        int64_t current_offset = index_size;

        // Build index
        for(auto& state : states)
        {
            IndexEntry e;
            e.round  = state.round;
            e.seed   = state.seed;
            e.year   = state.year;
            e.offset = current_offset;
            e.size   = calc_record_size(state);
            current_offset += e.size;
            index.push_back(e);
        }

        // Write file
        std::ofstream f(path, std::ios::binary);
        if(!f) { printf("Could not create %s\n", path.c_str()); return; }

        write_index(f);

        for(auto& state : states)
        {
            write_int(f, state.round);
            write_int(f, state.seed);
            write_int(f, state.year);
            write_int(f, state.width);
            write_int(f, state.height);
            write_grid(f, state.grid);
            write_int(f, state.settlements.size());
            for(auto& s : state.settlements)
                write_settlement(f, s);
        }

        f.close();
        printf("Database built: %d records saved to %s\n", 
            (int)states.size(), path.c_str());
    }

    // ─── Load index ──────────────────────────────────────────────────────────
    void load_index()
    {
        std::ifstream f(path, std::ios::binary);
        if(!f) { printf("Could not open %s\n", path.c_str()); return; }
        read_index(f);
        printf("Index loaded: %d records in database\n", (int)index.size());
    }

    // ─── Get full world state ────────────────────────────────────────────────
    WorldState get(int round, int seed, int year)
    {
        IndexEntry* e = find_entry(round, seed, year);
        if(!e)
        {
            printf("Record not found: round=%d seed=%d year=%d\n", round, seed, year);
            return WorldState();
        }

        std::ifstream f(path, std::ios::binary);
        f.seekg(e->offset);

        WorldState state;
        state.round  = read_int(f);
        state.seed   = read_int(f);
        state.year   = read_int(f);
        state.width  = read_int(f);
        state.height = read_int(f);
        state.grid   = read_grid(f);

        int num_settlements = read_int(f);
        for(int i = 0; i < num_settlements; i++)
            state.settlements.push_back(read_settlement(f));

        return state;
    }

    // ─── Get a single cell ───────────────────────────────────────────────────
    int get_cell(int round, int seed, int year, int x, int y)
    {
        WorldState state = get(round, seed, year);
        if(state.grid.empty()) return -1;
        return state.grid[y][x];
    }

    // ─── Get all settlements ─────────────────────────────────────────────────
    std::vector<Settlement> get_settlements(int round, int seed, int year)
    {
        return get(round, seed, year).settlements;
    }

    // ─── Get only alive settlements ──────────────────────────────────────────
    std::vector<Settlement> get_alive_settlements(int round, int seed, int year)
    {
        std::vector<Settlement> alive;
        for(auto& s : get_settlements(round, seed, year))
            if(s.alive) alive.push_back(s);
        return alive;
    }

    // ─── Get only port settlements ───────────────────────────────────────────
    std::vector<Settlement> get_ports(int round, int seed, int year)
    {
        std::vector<Settlement> ports;
        for(auto& s : get_settlements(round, seed, year))
            if(s.has_port) ports.push_back(s);
        return ports;
    }

    // ─── Get cells of a specific terrain type ────────────────────────────────
    std::vector<std::pair<int,int>> get_cells_of_type(int round, int seed, int year, int terrain)
    {
        std::vector<std::pair<int,int>> cells;
        WorldState state = get(round, seed, year);
        for(int y = 0; y < state.height; y++)
            for(int x = 0; x < state.width; x++)
                if(state.grid[y][x] == terrain)
                    cells.push_back({x, y});
        return cells;
    }

    // ─── Get all years available for a round+seed ────────────────────────────
    std::vector<int> get_years(int round, int seed)
    {
        std::vector<int> years;
        for(auto& e : index)
            if(e.round == round && e.seed == seed)
                years.push_back(e.year);
        return years;
    }

    // ─── Get all seeds for a round ───────────────────────────────────────────
    std::vector<int> get_seeds(int round)
    {
        std::vector<int> seeds;
        for(auto& e : index)
        {
            bool found = false;
            for(int s : seeds) if(s == e.seed) found = true;
            if(!found && e.round == round) seeds.push_back(e.seed);
        }
        return seeds;
    }

    // ─── Get all rounds ──────────────────────────────────────────────────────
    std::vector<int> get_rounds()
    {
        std::vector<int> rounds;
        for(auto& e : index)
        {
            bool found = false;
            for(int r : rounds) if(r == e.round) found = true;
            if(!found) rounds.push_back(e.round);
        }
        return rounds;
    }

    // ─── Get all records ──────────────────────────────────────────────────────
    std::vector<WorldState> get_all()
    {
        std::vector<WorldState> states;
        for(auto& e : index)
            states.push_back(get(e.round, e.seed, e.year));
        return states;
    }

    // ─── Check if a record exists ───────────────────────────────────────────
    bool has(int round, int seed, int year)
    {
        return find_entry(round, seed, year) != nullptr;
    }

    // ─── Print summary ───────────────────────────────────────────────────────
    void print_summary()
    {
        printf("\n--- Database Summary: %s ---\n", path.c_str());
        printf("%-8s %-8s %-8s %-12s %-10s\n", 
            "Round", "Seed", "Year", "Offset", "Size");
        for(auto& e : index)
            printf("%-8d %-8d %-8d %-12lld %-10d\n", 
                e.round, e.seed, e.year, e.offset, e.size);
        printf("Total records: %d\n\n", (int)index.size());
    }
};