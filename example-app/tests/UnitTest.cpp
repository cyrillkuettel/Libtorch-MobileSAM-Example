
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

// Function to read a binary file into a vector of unsigned char
// Function to read a binary file into a vector of unsigned char
std::vector<unsigned char> readFileToBuffer(const fs::path& filePath) {
    std::ifstream file(filePath, std::ios::binary | std::ios::ate);
    if (!file.is_open()) {
        throw std::runtime_error("Failed to open the file");
    }

    std::streamsize fileSize = file.tellg();
    file.seekg(0, std::ios::beg);

    std::vector<unsigned char> buffer(fileSize);
    if (!file.read(reinterpret_cast<char*>(buffer.data()), fileSize)) {
        throw std::runtime_error("Failed to read the file");
    }

    return buffer;
}
