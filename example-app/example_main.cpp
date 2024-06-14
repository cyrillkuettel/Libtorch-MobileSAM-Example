// Created by cyrill on 09.06.2024.
//

#include <iostream>  // todo: remove?

#include "mobile_sam.hpp"

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
    "images/picture2.jpg",
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
        const AppConfig config = {
            {{130.0f,80.0f}},
            {1.0f},  // top left, bottom right
            "images/elephants.jpg",
            false,
        };

        validateAppConfig(config);

        // Get the directory of the current source file
        fs::path sourceDir = fs::path(__FILE__).parent_path();
        std::cout << "Source directory: " << sourceDir << std::endl;
        assert(fs::exists(sourceDir));
        assert(fs::exists(sourceDir / "images") && "The images directory does not exist");

        // Construct the relative paths based on the source directory
        fs::path defaultImagePath = sourceDir / config.defaultImagePath;
        fs::path defaultMobileSamPredictor = sourceDir / "models" / "mobilesam_predictor.pt";
        fs::path defaultVitImageEmbedding = sourceDir / "models" / "vit_image_embedding.pt";
        fs::path yoloModelPath = sourceDir / "models" / "yolov5s.torchscript.ptl";

        SamPredictor predictor(1024, defaultMobileSamPredictor.string(),
                               defaultVitImageEmbedding.string());

        torch::Tensor maskInput =
            torch::zeros({1, 1, 256, 256}, torch::dtype(torch::kFloat32));

        cv::Mat jpg = cv::imread(defaultImagePath.string(), cv::IMREAD_COLOR);
        std::cout << "Reading image from: " << defaultImagePath << std::endl;

        if (jpg.channels() != 3) {
                std::cerr << "Input is not a 3-channel image" << std::endl;
                return 1;
        }
        predictor.setImage(jpg);

        auto clonedConfig = config;  // Clone the config struct
        std::vector<std::pair<float, float>> points;

        auto [pointCoordsTensor, pointLabelsTensor] =
            computePointsAndLabels(clonedConfig, jpg, predictor, yoloModelPath);
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

        return 0;
}
