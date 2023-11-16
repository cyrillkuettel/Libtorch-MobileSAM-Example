//
// Created by cyrill on 16.11.23.
//

#ifndef EXAMPLE_APP_PREDICTOR_H
#define EXAMPLE_APP_PREDICTOR_H


#ifndef SAM_PREDICTOR_H
#define SAM_PREDICTOR_H

#include <iostream>
#include <memory>
#include <vector>
#include <stdexcept>
#include <string>
#include <sstream>
#include <stdlib.h>

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgcodecs/imgcodecs.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/opencv.hpp>

#include <torch/script.h>
#include <torch/torch.h>
#include "resize_longest_size.h"

class SamPredictor {
public:
    SamPredictor(int target_length, const std::string &predictor_model_path,
                 const std::string &image_embedding_model_path)
            : resizeLongestSide(target_length),
              is_image_set(false), originalWidth(0), originalHeight(0) {
        reset_image();
        predictorModel = torch::jit::load(predictor_model_path);
        imageEmbeddingModel = torch::jit::load(image_embedding_model_path);
    }


    void set_image(const cv::Mat &image);

private:

    torch::jit::script::Module predictorModel;
    torch::jit::script::Module imageEmbeddingModel;
    ResizeLongestSide resizeLongestSide;
    torch::Tensor features;
    bool is_image_set;
    int originalWidth;
    int originalHeight;

    const std::vector<float> pixel_mean = {123.675, 116.28, 103.53};
    const std::vector<float> pixel_std = {58.395, 57.12, 57.375};

    const std::pair<int, int> input_size = {1024, 1024};

    void set_torch_image(torch::Tensor &inputTensor);

    void reset_image();

    std::tuple<torch::Tensor, torch::Tensor, torch::Tensor>
    predict(const torch::Tensor &pointCoords, const torch::Tensor &pointLabels);

};

#endif // SAM_PREDICTOR_H
#endif //EXAMPLE_APP_PREDICTOR_H
