#ifndef MAIN_H
#define MAIN_H

#include <chrono>
#include "predictor.h"
#include "visualize.hpp"
#include "yolo.h"

#include <iostream>
#include <vector>
#include <filesystem>

namespace fs = std::filesystem;

// Main api for the outside world

struct AppConfig {
        std::vector<std::pair<float, float>> points;
        std::vector<float> pointLabels;
        std::string defaultImagePath;
        bool useYoloBoxes;
};

std::pair<torch::Tensor, torch::Tensor> computePointsAndLabels(
    const AppConfig& config, cv::Mat& jpg, SamPredictor& predictor, const fs::path& yoloModelPath);



#endif // MAIN_H
