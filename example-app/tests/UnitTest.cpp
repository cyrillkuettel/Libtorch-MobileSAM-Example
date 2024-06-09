
#include "UnitTest.h"


// This file will basically function as a utils file in a way.
SamPredictor setupPredictor() {
        // test here

        // Get the directory of the current source file
        fs::path sourceDir = fs::path(__FILE__).parent_path().parent_path();

        std::cout << "Source directory: " << sourceDir << std::endl;
        assert(fs::exists(sourceDir));
        assert(fs::exists(sourceDir / "images") && "The images directory does not exist");
        // Construct the relative paths based on the source directory

        fs::path defaultMobileSamPredictor = sourceDir / "models" / "mobilesam_predictor.pt";
        fs::path defaultVitImageEmbedding = sourceDir / "models" / "vit_image_embedding.pt";
        fs::path yoloModelPath = sourceDir / "models" / "yolov5s.torchscript.ptl";
        SamPredictor predictor(1024, defaultMobileSamPredictor.string(),
                               defaultVitImageEmbedding.string());
        return predictor;
}

