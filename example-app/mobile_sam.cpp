
#include "mobile_sam.hpp"

namespace fs = std::filesystem;


void validateAppConfig(const AppConfig& config) {
        if (config.useYoloBoxes) {
                if (!config.points.empty() || !config.pointLabels.empty()) {
                        std::cout << "AppConfig is invalid. Note that if  "
                                     "config.useYoloBoxes==true, "
                                     "the boxes will be calculated using the "
                                     "YOLOV5 object detection model.\n";
                        std::exit(EXIT_FAILURE);
                }
        }
        std::cout << "AppConfig ok.\n";
}


// todo: break up into 2 functions, so that we can use the points before transformation is applied, this can be used for showing the points on the original image.
std::pair<torch::Tensor, torch::Tensor> computePointsAndLabels(
    const AppConfig& config, cv::Mat& jpg, SamPredictor& predictor, const fs::path& yoloModelPath) {

        std::vector<std::pair<float, float>> points;
        if (config.useYoloBoxes) {
                std::cout << "Using Yolo Boxes" << std::endl;
                runYolo(jpg, points, yoloModelPath);  // writes into the points
        } else {

                std::cout << "Using config points=" << config.points
                          << std::endl;
                points = config.points;
        }

        // `pointLabels`: Labels for the sparse input prompts.
        // 0 is a negative input point,
        // 1 is a positive input point,
        // 2 is a top-left box corner,
        // 3 is a bottom-right box corner,
        // and -1 is a padding point. If there is no box input,
        // a single padding point with label -1 and coordinates (0.0, 0.0) should be concatenated.
        std::vector<float> pointLabels;

        if (!config.useYoloBoxes) {
                if (config.pointLabels.empty()) {
                        pointLabels = {1.0f};
                } else {
                        pointLabels = config.pointLabels;
                }
        } else {
                if (points.size() == 4) {
                        pointLabels = {2.0f, 3.0f, 2.0f, 3.0f};
                } else if (points.size() == 2) {
                        pointLabels = {2.0f, 3.0f};
                } else {
                        std::cerr << "Invalid number of points" << std::endl;
                        std::exit(EXIT_FAILURE);
                }
        }

        // pad
        assert(!pointLabels.empty());
        while (pointLabels.size() < 5) {
                pointLabels.emplace_back(-1.0f);
        }

        assert(!points.empty());
        while (points.size() < 5) {
                points.emplace_back(0.0f, 0.0f);
        }
        std::vector<std::pair<float, float>> transformedCoords =
            predictor.transform.applyCoords(
                points,
                {predictor.originalImageHeight, predictor.originalImageWidth});

        // Convert the transformed coordinates back to a flat vector
        std::vector<float> flatTransformedCoords;
        for (const auto& [first, second] : transformedCoords) {
                flatTransformedCoords.push_back(first);
                flatTransformedCoords.push_back(second);
        }
        std::cout << "flatTransformedCoords" << std::endl;
        std::cout << flatTransformedCoords << std::endl;
        assert(flatTransformedCoords.size() == 10);
        assert(pointLabels.size() == 5);

        std::cout << "pointLabels.size(): " << pointLabels.size() << std::endl;
        std::cout << pointLabels << std::endl;

        torch::Tensor pointCoordsTensor =
            torch::tensor(flatTransformedCoords, torch::dtype(torch::kFloat32));

        torch::Tensor pointLabelsTensor =
            torch::tensor(pointLabels, torch::dtype(torch::kFloat32));

        return std::make_pair(pointCoordsTensor, pointLabelsTensor);
}
