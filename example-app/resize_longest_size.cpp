//
// Created by cyrill on 16.11.23.
//

#include "resize_longest_size.h"

#include <algorithm>
#include <cmath>

ResizeLongestSide::ResizeLongestSide(int target_length)
    : target_length(target_length) {}

std::pair<int, int> ResizeLongestSide::getPreprocessShape(int oldh,
                                                          int oldw) const {
        double scale =
            static_cast<double>(target_length) / std::max(oldh, oldw);
        int newh = std::round(oldh * scale);
        int neww = std::round(oldw * scale);
        return {newh, neww};
}

void ResizeLongestSide::applyImage(cv::Mat& image) {
        auto [newh, neww] = getPreprocessShape(image.rows, image.cols);

        // print these out
        std::cout << "newh: " << newh << std::endl;
        std::cout << "neww: " << neww << std::endl;

        cv::resize(image, image, cv::Size(neww, newh));

        std::cout << "image.cols: " << image.cols << std::endl;
        std::cout << "image.rows: " << image.rows << std::endl;
}
std::vector<std::pair<float, float>> ResizeLongestSide::applyCoords(
    const std::vector<std::pair<float, float>>& coords,
    const std::pair<int, int>& original_size) const {
        /**
         * Transforms coordinates based on the resizing of an image.
         *
         * The function takes a vector of coordinates (each as a pair of floats) and the original
         * dimensions of the image (as a pair of ints). Each coordinate is scaled according to
         * the transformation applied to the image when resizing its longest side to the target length.
         *
         */

        int old_h = original_size.first;
        int old_w = original_size.second;
        auto [new_h, new_w] = getPreprocessShape(old_h, old_w);

        std::vector<std::pair<float, float>> transformed_coords;
        for (const auto& coord : coords) {
                float transformed_x =
                    coord.first * (static_cast<float>(new_w) / old_w);
                float transformed_y =
                    coord.second * (static_cast<float>(new_h) / old_h);
                transformed_coords.emplace_back(transformed_x, transformed_y);
        }

        return transformed_coords;
}