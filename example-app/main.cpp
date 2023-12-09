#include <chrono>
#include "predictor.h"
#include "visualize.cpp"
#include "yolo.h"

struct AppConfig {
        std::vector<std::pair<float, float>> points = {};
        std::vector<float> pointLabels = {};
        std::string defaultImagePath;
        bool useYoloBoxes = false;
};
void validateAppConfig(const AppConfig& config) {
    if (config.useYoloBoxes) {
        if (!config.points.empty() || !config.pointLabels.empty()) {
                std::cout << "AppConfig is invalid. Note that if  config.useYoloBoxes==true, "
                             "the boxes will be calculated using the YOLOV5 object detection model.\n";
                std::exit(EXIT_FAILURE);
        }
    }
    std::cout << "AppConfig ok.\n";
}
const AppConfig exampleInputPackage = {
    {{228.0f, 102.0f}, {325.0f, 261.0f}},
    {2.0f, 3.0f},  // top left, bottom right
    //"/home/cyrill/pytorch/libtorch-opencv/example-app/images/img.jpg",
    "/Users/cyrill/Libtorch-MobileSAM-Example/example-app/images/img.jpg",
};

const AppConfig exampleInputPackage2 = {
    {
        {384.0f, 229.0f},
    },
    {1.0f},
    "/home/cyrill/pytorch/libtorch-opencv/example-app/images/picture2.jpg",
};

const AppConfig issue1Input = {
    {
        {460.0f, 200.0f},
        {790.0f, 592.0f},
    },
    {2.0f, 3.0f},  // top left, bottom right
    "/Users/cyrill/Desktop/dog.png",
};

const AppConfig iosTest = {
    {
        {460.0f, 200.0f},
        {790.0f, 592.0f},
    },
    {2.0f, 3.0f},  // top left, bottom right
    "/Users/cyrill/Downloads/test.jpg",
};

const AppConfig iosTestY = {
    {
    },
    {},  // top left, bottom right
    "/Users/cyrill/Downloads/test.jpg",
        true,
};

const AppConfig wald = {
    {
        {2286.0f, 336.0f},
    },
    {1.0f},  // top left, bottom right
    "/Users/cyrill/Desktop/wald.jpg",
};

const AppConfig mateY = {
    { },
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
    "/Users/cyrill/Libtorch-MobileSAM-Example/example-app/images/img.jpg",
    true,
};

int main() {
        // Set input package here
        const AppConfig config = iosTestY;
        validateAppConfig(config);

        std::string defaultImagePath = config.defaultImagePath;

        //	std::string defaultMobileSamPredictor =
        //		"/home/cyrill/pytorch/libtorch-opencv/example-app/models/mobilesam_predictor.pt";
        //	std::string defaultVitImageEmbedding =
        //		"/home/cyrill/pytorch/libtorch-opencv/example-app/models/vit_image_embedding.pt";

        std::string defaultMobileSamPredictor =
            "/Users/cyrill/Libtorch-MobileSAM-Example/example-app/models/"
            "mobilesam_predictor.pt";
        std::string defaultVitImageEmbedding =
            "/Users/cyrill/Libtorch-MobileSAM-Example/example-app/models/"
            "vit_image_embedding.pt";

        SamPredictor predictor(1024, defaultMobileSamPredictor,
                               defaultVitImageEmbedding);
        torch::Tensor maskInput =
            torch::zeros({1, 1, 256, 256}, torch::dtype(torch::kFloat32));

        cv::Mat jpg = cv::imread(defaultImagePath, cv::IMREAD_COLOR);
        if (jpg.channels() != 3) {
                std::cerr << "Input is not a 3-channel image" << std::endl;
                return 1;
        }
        predictor.setImage(jpg);

        std::vector<std::pair<float, float> > points;
        if (config.useYoloBoxes) {
                std::cout << "Using Yolo Boxes" << std::endl;
                runYolo(jpg, points); // writes into the points
        } else {
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
                pointLabels = config.pointLabels;
        } else {
                if (points.size() == 4) {
                        pointLabels = {2.0f, 3.0f, 2.0f, 3.0f};
                } else if (points.size() == 2) {
                        pointLabels = {2.0f, 3.0f};
                } else {
                        std::cerr << "Invalid number of points" << std::endl;
                        return 1;
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
        for (const auto& coord : transformedCoords) {
                flatTransformedCoords.push_back(coord.first);
                flatTransformedCoords.push_back(coord.second);
        }

        std::cout << flatTransformedCoords << std::endl;
        assert(flatTransformedCoords.size() == 10);
        assert(pointLabels.size() == 5);

        std::cout << "pointLabels.size(): " << pointLabels.size() << std::endl;
        std::cout << pointLabels << std::endl;

        torch::Tensor pointCoordsTensor =
            torch::tensor(flatTransformedCoords, torch::dtype(torch::kFloat32));

        torch::Tensor pointLabelsTensor =
            torch::tensor(pointLabels, torch::dtype(torch::kFloat32));

        pointCoordsTensor =
            pointCoordsTensor.reshape({1, 5, 2}).to(torch::kFloat32);
        pointLabelsTensor =
            pointLabelsTensor.reshape({1, 5}).to(torch::kFloat32);
        bool hasMaskInput = false;
        /***
	 * run inference
	 */
        auto start = std::chrono::high_resolution_clock::now();
        torch::Tensor masks, IOUPredictions, lowResMasks;
        std::tie(masks, IOUPredictions, lowResMasks) = predictor.predict(
            pointCoordsTensor, pointLabelsTensor, maskInput, hasMaskInput);
        // Stop timing after the first function call
        auto stop = std::chrono::high_resolution_clock::now();
        auto duration =
            std::chrono::duration_cast<std::chrono::milliseconds>(stop - start);
        std::cout << "predictor.predict duration: " << duration.count()
                  << "ms\n";

        visualizeResults(jpg, masks, IOUPredictions, pointCoordsTensor);
}