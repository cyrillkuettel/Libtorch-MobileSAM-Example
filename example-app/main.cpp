#include <chrono>
#include "predictor.h"
#include "visualize.cpp"
#include "yolo.h"

#include <iostream>
#include <vector>
#include <filesystem>

namespace fs = std::filesystem;

bool mouseClicked = false;

static void mouseCallback(int event, int x, int y, int flags, void* userdata) {
        // Check if the left mouse button was clicked
        if (event == cv::EVENT_LBUTTONDOWN) {

                std::cout << "LBUTTONDOWN\n";
                // Cast userdata to the expected type (std::vector<std::pair<int, int>>*)
                auto* clickedPoints =
                    static_cast<std::vector<std::pair<int, int>>*>(userdata);
                clickedPoints->emplace_back(x, y);
                std::cout << "Writing into memory points: " << clickedPoints
                          << '\n';
                // Optional: Print the coordinates for verification
                std::cout << "Clicked at: (" << x << ", " << y << ")\n";
                mouseClicked = true;
        }
        // cv::destroyWindow("Select a point"); // Close the window
}

struct AppConfig {
        std::vector<std::pair<float, float>> points = {};
        std::vector<float> pointLabels = {};
        std::string defaultImagePath;
        bool useYoloBoxes = false;
};

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

std::pair<torch::Tensor, torch::Tensor> computePointsAndLabels(
    const AppConfig& config, cv::Mat& jpg, SamPredictor& predictor) {

        std::vector<std::pair<float, float>> points;
        if (config.useYoloBoxes) {
                std::cout << "Using Yolo Boxes" << std::endl;
                runYolo(jpg, points);  // writes into the points
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

const AppConfig exampleInputPackage = {
    {{228.0f, 102.0f}, {325.0f, 261.0f}},
    {2.0f, 3.0f},  // top left, bottom right
    //"/home/cyrill/pytorch/libtorch-opencv/example-app/images/img.jpg",
    "images/img.jpg",
};

const AppConfig exampleInputPackage2 = {
    {
        {384.0f, 229.0f},
    },
    {1.0f},
    "/home/cyrill/pytorch/libtorch-opencv/example-app/images/picture2.jpg",
};


const AppConfig wald = {
    {
        {2286.0f, 336.0f},
    },
    {1.0f},  // top left, bottom right
    "/Users/cyrill/Desktop/wald.jpg",
};

const AppConfig mateY = {
    {},
    {},  // top left, bottom right
    "/Users/cyrill/Documents/Test_images/mate.jpg",
    true,
};

const AppConfig waldY = {
    {},
    {},  // top left, bottom right
    "/Users/cyrill/Desktop/wald.jpg",
    true,
};

const AppConfig exampleInputPackageY = {
    {},
    {},  // top left, bottom right
    "images/img.jpg",
    true,
};
const AppConfig elephantsY = {

    {},
    {},  // top left, bottom right
    "images/elephants.jpg",
    true,
};

const AppConfig test = {
    {},
    {},  // top left, bottom right
    "images/picture2.jpg",
    false,
};

const AppConfig test2 = {
    {{420.0f, 600.0f}},
    {1.0f},  // top left, bottom right
    "/Users/cyrill/Desktop/nuessli2.jpeg",
    false,
};

int main() {
        const AppConfig config = elephantsY;
        validateAppConfig(config);

        // Get the current working directory
        fs::path currentPath = fs::current_path();

        // Construct the relative paths based on the current directory
        fs::path defaultImagePath =
            currentPath / "images" /
            fs::path(config.defaultImagePath).filename();
        fs::path defaultMobileSamPredictor =
            currentPath /  "models" / "mobilesam_predictor.pt";
        fs::path defaultVitImageEmbedding =
            currentPath / "models" / "vit_image_embedding.pt";

        SamPredictor predictor(1024, defaultMobileSamPredictor.string(),
                               defaultVitImageEmbedding.string());
        torch::Tensor maskInput =
            torch::zeros({1, 1, 256, 256}, torch::dtype(torch::kFloat32));

        cv::Mat jpg = cv::imread(defaultImagePath.string(), cv::IMREAD_COLOR);
        if (jpg.channels() != 3) {
                std::cerr << "Input is not a 3-channel image" << std::endl;
                return 1;
        }
        predictor.setImage(jpg);

        auto clonedConfig = config;  // Clone the config struct
        std::vector<std::pair<float, float>> points;
        if (!config.useYoloBoxes) {
                cv::namedWindow("Select a point", cv::WINDOW_AUTOSIZE);

                std::cout << "setting mouseCallback" << std::endl;
                cv::imshow("Select a point", jpg);
                cv::setMouseCallback("Select a point", mouseCallback, &points);

                while (true) {
                        if (cv::waitKey(1) > 0 || mouseClicked) {
                                break;
                        }
                }

                if (mouseClicked) {
                        std::cout << "mouseCallback was invoked." << std::endl;
                } else {
                        std::cout << "!!!! mouseCallback was not invoked."
                                  << std::endl;
                        std::exit(EXIT_FAILURE);
                }

                assert(!points.empty());
                clonedConfig.points = points;
        }

        auto [pointCoordsTensor, pointLabelsTensor] =
            computePointsAndLabels(clonedConfig, jpg, predictor);
        pointCoordsTensor =
            pointCoordsTensor.reshape({1, 5, 2}).to(torch::kFloat32);
        pointLabelsTensor =
            pointLabelsTensor.reshape({1, 5}).to(torch::kFloat32);
        bool hasMaskInput = false;

        auto start = std::chrono::high_resolution_clock::now();
        torch::Tensor masks;
        torch::Tensor IOUPredictions;
        torch::Tensor lowResMasks;
        std::tie(masks, IOUPredictions, lowResMasks) = predictor.predict(
            pointCoordsTensor, pointLabelsTensor, maskInput, hasMaskInput);
        auto stop = std::chrono::high_resolution_clock::now();
        auto duration =
            std::chrono::duration_cast<std::chrono::milliseconds>(stop - start);
        std::cout << "predictor.predict duration: " << duration.count()
                  << "ms\n";

        visualizeResults(jpg, masks, IOUPredictions, pointCoordsTensor);
}
