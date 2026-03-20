#pragma once
#include <vector>
#include <string>
#include <fstream>
#include <cstdint>

struct Round
{
    std::string id;
    int round_number;
    std::string event_date;
    std::string status;
    int map_width;
    int map_height;
    int seeds_count;
    float round_weight;
    std::string started_at;
    std::string closes_at;
    int prediction_window_minutes;
};

struct Budget
{
    std::string round_id;
    int queries_used;
    int queries_max;
    bool active;
};

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

struct InitialState
{
    int seed_index;
    std::vector<std::vector<int>> grid;
    std::vector<Settlement> settlements; 
};

struct SimulateResponse
{
    std::vector<std::vector<int>> grid;
    std::vector<Settlement> settlements;
    int viewport_x;
    int viewport_y;
    int viewport_w;
    int viewport_h;
    int width; 
    int height;
    int queries_used;
    int queries_max;
};

struct MyPrediction
{
    int seed_index;
    std::vector<std::vector<int>> argmax_grid;
    std::vector<std::vector<float>> confidence_grid;
    float score;
    std::string submitted_at;
};

struct Analysis
{
    std::vector<std::vector<std::vector<float>>> prediction; 
    std::vector<std::vector<std::vector<float>>> ground_truth; 
    float score;
    int width;
    int height;
    std::vector<std::vector<int>> initial_grid; 
};

struct SeedObservations
{
    int seed_index;
    int map_width;
    int map_height;
    std::vector<std::vector<int>> initial_grid;
    std::vector<std::vector<int>> observed_grid;
    std::vector<std::vector<int>> observation_count;
    std::vector<std::vector<std::vector<int>>> class_counts;

    static int terrain_to_class(int terrain)
    {
        switch(terrain)
        {
            case 0:  return 0; // Empty
            case 10: return 0; // Ocean
            case 11: return 0; // Plains
            case 1:  return 1; // Settlement
            case 2:  return 2; // Port
            case 3:  return 3; // Ruin
            case 4:  return 4; // Forest
            case 5:  return 5; // Mountain
            default: return 0;
        }
    }

    void record_observation(int x, int y, int terrain)
    {
        observed_grid[y][x] = terrain;
        observation_count[y][x]++;
        class_counts[y][x][terrain_to_class(terrain)]++;
    }
};