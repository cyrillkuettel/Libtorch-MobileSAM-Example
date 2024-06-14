//
// Created by cyrill on 09.06.2024.
//

#ifndef VISUALIZE_H
#define VISUALIZE_H

#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgcodecs/imgcodecs.hpp>

#include <opencv2/core.hpp>  // For basic OpenCV structures (cv::Mat, cv::Vec3b, etc.)
#include <opencv2/imgproc.hpp>  // For image processing functions like cv::resize
#include <opencv2/opencv.hpp>

#include <torch/script.h>

void printMatType(const cv::Mat& mat);

cv::Mat tensorToMat(torch::Tensor tensor);

void showMask(const torch::Tensor& mask, cv::Mat& image);

void showPoints(const torch::Tensor& coords, cv::Mat& image,
                int markerSize = 6);

void visualizeResults(cv::Mat& image, const torch::Tensor& masks,
                      const torch::Tensor& scores,
                      const torch::Tensor& pointCoords);

std::vector<cv::Mat> createInMemoryImages(cv::Mat& image,
                                          const torch::Tensor& masks,
                                          const torch::Tensor& scores,
                                          const torch::Tensor& pointCoords);

void saveAndDisplayImages(
    const std::vector<std::vector<unsigned char>>& inMemoryImages,
    const torch::Tensor& scores);

#endif  // VISUALIZE_H
