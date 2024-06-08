//
// Created by cyrill on 09.12.23.
//

#include "yolo.h"

// the caller is responsible for deleting the returned buffer
float* resizeAndNormalizeImage(cv::Mat& inputImage, int w, int h) {
        cv::Mat resizedImage;
        cv::resize(inputImage, resizedImage, cv::Size(w, h));

        int bSize = 3 * h * w;
        float* normalizedBuffer = new float[bSize];
        std::cout << "Normalized buffer size: " << bSize << std::endl;

        for (int i = 0; i < (w * h); i++) {
                cv::Vec3b pixel = resizedImage.at<cv::Vec3b>(i / w, i % w);

                normalizedBuffer[i] = pixel[0] / 255.0f;  // Blue channel
                normalizedBuffer[w * h + i] =
                    pixel[1] / 255.0f;  // Green channel
                normalizedBuffer[w * h * 2 + i] =
                    pixel[2] / 255.0f;  // Red channel
        }

        return normalizedBuffer;
}

/**
 * Extracts the top bounding boxes for segmenting objects from an output tensor array.
 * The filters the boxes based on their scores, retaining only the most probable ones.
 */
void getBestBoxes(float *outputTensorFloatArray, int32_t inputWidth, int32_t inputHeight,
                  int32_t imageWidth, int32_t imageHeight,  int outputRows, int outputColumns,
                  std::vector<std::pair<float, float>>& points)
{
        float probThreshold = 0.3f;

        float imgScaleX = (float)imageWidth / inputWidth;
        float imgScaleY = (float)imageHeight / inputHeight;
        float ivScaleX = 1.0f;
        float ivScaleY = 1.0f;

        std::vector<std::pair<float, std::pair<float, float>>> objectsWithScores;

        for (int i = 0; i < outputRows; i++) {
                float score = outputTensorFloatArray[i * outputColumns + 4];
                if (score > probThreshold) {

                        float x = outputTensorFloatArray[i * outputColumns];
                        float y = outputTensorFloatArray[i * outputColumns + 1];
                        float w = outputTensorFloatArray[i * outputColumns + 2];
                        float h = outputTensorFloatArray[i * outputColumns + 3];

                        float left = imgScaleX * (x - w / 2);
                        float top = imgScaleY * (y - h / 2);
                        float right = imgScaleX * (x + w / 2);
                        float bottom = imgScaleY * (y + h / 2);

                        float rectLeft = ivScaleX * left;
                        float rectTop = top * ivScaleY;
                        float rectRight = ivScaleX * right;
                        float rectBottom = ivScaleY * bottom;

                        objectsWithScores.push_back({score, {rectLeft, rectTop}});
                        objectsWithScores.push_back({score, {rectRight, rectBottom}});
                }
        }

        std::sort(objectsWithScores.begin(), objectsWithScores.end(),
                  [](const auto& a, const auto& b) { return a.first > b.first; });


        // limit points. 4 points means 2 boxes.
        // More is not possible, this is a limitation of the SamOnnxModel
        // on which TorchScripted mobilesam_predictor.pt is based on.
        // You might be able to use more points with the full model.
        for (int i = 0; i < 2 && i < objectsWithScores.size(); ++i) {
                points.push_back(objectsWithScores[i].second);
        }
        std::cout << "Points that have been selected" << std::endl;
        for (const auto& point : points) {
                std::cout << "Point: (x:" << point.first << ", y:" << point.second << ")\n";
        }
        assert(points.size() == 4 || points.size() == 2);
}

void runYolo(cv::Mat& inputImage, std::vector<std::pair<float, float>>&  points) {
        int32_t originalImageWidth = inputImage.rows;
        int32_t originalImageHeight = inputImage.cols;

        // get boxes from yolo .
        torch::jit::script::Module _impl = torch::jit::load(
            "/Users/cyrill/libtorch-mobileSAM-exapmle/Libtorch-MobileSAM-Example/example-app/models/"
            "yolov5s.torchscript.ptl");

        float mean[3] = {0.0, 0.0, 0.0};  // Yolov5s mobile scripted mean
        float std[3] = {1.0, 1.0,
                        1.0};  // Yolov5s mobile scripted standard deviation
        int32_t yoloInputWidth = 640;
        int32_t yoloInputHeight = 640;

        float* buffer = resizeAndNormalizeImage(inputImage, yoloInputWidth, yoloInputHeight);

        std::vector<torch::jit::IValue> inputs;
        at::Tensor outputTensor;
        auto inputTensor = torch::from_blob(
            buffer, {1, 3, yoloInputWidth, yoloInputHeight}, at::kFloat);
        inputs.emplace_back(inputTensor);

        // run yolo inference
        auto outputTuple = _impl.forward(inputs).toTuple();
        outputTensor = outputTuple->elements()[0].toTensor();

        float* floatBuffer = outputTensor.data_ptr<float>();
        delete[] buffer;

        int outputRows = outputTensor.sizes()[1];
        int outputColumns = outputTensor.sizes()[2];

        const char* constYoloClassNames[] = {
            "person",        "bicycle",      "car",
            "motorcycle",    "airplane",     "bus",
            "train",         "truck",        "boat",
            "traffic light", "fire hydrant", "stop sign",
            "parking meter", "bench",        "bird",
            "cat",           "dog",          "horse",
            "sheep",         "cow",          "elephant",
            "bear",          "zebra",        "giraffe",
            "backpack",      "umbrella",     "handbag",
            "tie",           "suitcase",     "frisbee",
            "skis",          "snowboard",    "sports ball",
            "kite",          "baseball bat", "baseball glove",
            "skateboard",    "surfboard",    "tennis racket",
            "bottle",        "wine glass",   "cup",
            "fork",          "knife",        "spoon",
            "bowl",          "banana",       "apple",
            "sandwich",      "orange",       "broccoli",
            "carrot",        "hot dog",      "pizza",
            "donut",         "cake",         "chair",
            "couch",         "potted plant", "bed",
            "dining table",  "toilet",       "tv",
            "laptop",        "mouse",        "remote",
            "keyboard",      "cell phone",   "microwave",
            "oven",          "toaster",      "sink",
            "refrigerator",  "book",         "clock",
            "vase",          "scissors",     "teddy bear",
            "hair drier",    "toothbrush"};
        const size_t numStrings =
            sizeof(constYoloClassNames) / sizeof(constYoloClassNames[0]);

        char* yoloClassNames[numStrings];

        for (size_t i = 0; i < numStrings; ++i) {
                yoloClassNames[i] = new char[strlen(constYoloClassNames[i]) +
                                             1];  // +1 for null terminator
                strcpy(yoloClassNames[i], constYoloClassNames[i]);
        }

        getBestBoxes(floatBuffer,
                     yoloInputWidth,
                     yoloInputHeight,
                     originalImageWidth,
                     originalImageHeight,
                     outputRows,
                     outputColumns,
                     points);

}
