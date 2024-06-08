// In your header file (.h)
#ifndef YOLO_H
#define YOLO_H

#include <torch/script.h>  // For PyTorch's JIT compiler and model loading
   // For PyTorch's tensor operations
#include <cstring>         // For strlen and strcpy
#include <iostream>        // For standard I/O operations (std::cerr, std::endl)
#include <opencv2/core.hpp>  // For basic OpenCV structures (cv::Mat, cv::Vec3b, etc.)
#include <opencv2/imgproc.hpp>  // For image processing functions like cv::resize
               // For std::string
#include <vector>
#include <iostream>
#include <filesystem>
namespace fs = std::filesystem;

// Function declarations
float* resizeAndNormalizeImage(cv::Mat& inputImage, int w, int h);

void getBestBoxes(float *outputTensorFloatArray, int32_t inputWidth, int32_t inputHeight,
                  int32_t imageWidth, int32_t imageHeight,  int outputRows, int outputColumns,
                  std::vector<std::pair<float, float>>& points);

void runYolo(cv::Mat& inputImage, std::vector<std::pair<float, float>>& points, const fs::path& yoloModelPath);

#endif //YOLO_H
