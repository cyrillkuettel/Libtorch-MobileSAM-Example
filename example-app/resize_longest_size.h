//
// Created by cyrill on 16.11.23.
//

#ifndef EXAMPLE_APP_RESIZE_LONGEST_SIZE_H
#define EXAMPLE_APP_RESIZE_LONGEST_SIZE_H


#ifndef EXAMPLE_APP_TRANSFORMS_H
#define EXAMPLE_APP_TRANSFORMS_H

#include <utility>
#include <opencv2/opencv.hpp>

class ResizeLongestSide {
public:
    explicit ResizeLongestSide(int target_length);

    std::pair<int, int> get_preprocess_shape(int oldh, int oldw) const;
    void apply_image(cv::Mat &image);

private:
    int target_length_;
};

#endif //EXAMPLE_APP_TRANSFORMS_H


#endif //EXAMPLE_APP_RESIZE_LONGEST_SIZE_H
