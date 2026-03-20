#pragma once
#include <vector>
#include <cmath>
#include <cstdint>

static const int NUM_CLASSES = 6;

inline int terrain_to_class(int code)
{
    switch(code)
    {
        case 1: return 1;
        case 2: return 2;
        case 3: return 3;
        case 4: return 4;
        case 5: return 5;
        default: return 0; // 0, 10, 11 → class 0
    }
}

inline bool is_land(int code)
{
    return code != 10;
}

struct CellFeatures
{
    uint8_t x, y;
    uint8_t terrain_class;
    uint8_t is_land;

    uint8_t neighbors[8]; // N, NE, E, SE, S, SW, W, NW — class 0-5, 255 = OOB

    uint8_t adj_ocean;
    uint8_t adj_forest;
    uint8_t adj_mountain;
    uint8_t adj_settlement;
    uint8_t adj_plains;

    float dist_nearest_settlement;
    float dist_nearest_port;
    uint8_t settlements_r3;
    uint8_t settlements_r5;

    uint8_t is_coastal;
    float dist_coast;

    uint8_t land_cells_5x5;
};

static const int DX[8] = { 0,  1, 1, 1, 0, -1, -1, -1};
static const int DY[8] = {-1, -1, 0, 1, 1,  1,  0, -1};

inline std::vector<CellFeatures> extract_features(
    const std::vector<std::vector<int>>& grid)
{
    int H = grid.size();
    int W = H > 0 ? grid[0].size() : 0;

    std::vector<std::pair<int,int>> settlements;
    std::vector<std::pair<int,int>> ports;
    std::vector<std::pair<int,int>> oceans;

    for(int y = 0; y < H; y++)
    {
        for(int x = 0; x < W; x++)
        {
            int t = grid[y][x];
            if(t == 1) settlements.push_back({x, y});
            if(t == 2) { ports.push_back({x, y}); settlements.push_back({x, y}); }
            if(t == 10) oceans.push_back({x, y});
        }
    }

    std::vector<CellFeatures> features(H * W);

    for(int y = 0; y < H; y++)
    {
        for(int x = 0; x < W; x++)
        {
            CellFeatures& f = features[y * W + x];
            int code = grid[y][x];

            f.x = x;
            f.y = y;
            f.terrain_class = terrain_to_class(code);
            f.is_land = is_land(code) ? 1 : 0;

            f.adj_ocean = f.adj_forest = f.adj_mountain = 0;
            f.adj_settlement = f.adj_plains = 0;

            for(int d = 0; d < 8; d++)
            {
                int nx = x + DX[d];
                int ny = y + DY[d];
                if(nx < 0 || nx >= W || ny < 0 || ny >= H)
                {
                    f.neighbors[d] = 255;
                    continue;
                }
                int nc = terrain_to_class(grid[ny][nx]);
                f.neighbors[d] = nc;

                int nt = grid[ny][nx];
                if(nt == 10) f.adj_ocean++;
                if(nt == 4)  f.adj_forest++;
                if(nt == 5)  f.adj_mountain++;
                if(nt == 1 || nt == 2) f.adj_settlement++;
                if(nt == 11 || nt == 0) f.adj_plains++;
            }

            f.is_coastal = (f.is_land && f.adj_ocean > 0) ? 1 : 0;

            f.dist_nearest_settlement = 999.0f;
            f.settlements_r3 = 0;
            f.settlements_r5 = 0;
            for(auto& [sx, sy] : settlements)
            {
                float d = sqrtf((x-sx)*(x-sx) + (y-sy)*(y-sy));
                if(d < f.dist_nearest_settlement) f.dist_nearest_settlement = d;
                int md = abs(x-sx) + abs(y-sy);
                if(md <= 3) f.settlements_r3++;
                if(md <= 5) f.settlements_r5++;
            }
            if(settlements.empty()) f.dist_nearest_settlement = 99.0f;

            f.dist_nearest_port = 999.0f;
            for(auto& [px, py] : ports)
            {
                float d = sqrtf((x-px)*(x-px) + (y-py)*(y-py));
                if(d < f.dist_nearest_port) f.dist_nearest_port = d;
            }
            if(ports.empty()) f.dist_nearest_port = 99.0f;

            f.dist_coast = 999.0f;
            for(auto& [ox, oy] : oceans)
            {
                float d = sqrtf((x-ox)*(x-ox) + (y-oy)*(y-oy));
                if(d < f.dist_coast) f.dist_coast = d;
            }
            if(oceans.empty()) f.dist_coast = 99.0f;

            f.land_cells_5x5 = 0;
            for(int dy = -2; dy <= 2; dy++)
            {
                for(int dx = -2; dx <= 2; dx++)
                {
                    int nx = x + dx, ny = y + dy;
                    if(nx >= 0 && nx < W && ny >= 0 && ny < H && is_land(grid[ny][nx]))
                        f.land_cells_5x5++;
                }
            }
        }
    }

    return features;
}
