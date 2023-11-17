//
// Created by cyrill on 16.11.23.
//

#include "resize_longest_size.h"

#include <algorithm>
#include <cmath>

ResizeLongestSide::ResizeLongestSide(int target_length)
	: target_length_(target_length)
{
}

std::pair<int, int> ResizeLongestSide::get_preprocess_shape(int oldh,
							    int oldw) const
{
	double scale =
		static_cast<double>(target_length_) / std::max(oldh, oldw);
	int newh = std::round(oldh * scale);
	int neww = std::round(oldw * scale);
	return { newh, neww };
}

void ResizeLongestSide::apply_image(cv::Mat &image)
{
	auto [newh, neww] = get_preprocess_shape(image.rows, image.cols);

	// print these out
	std::cout << "newh: " << newh << std::endl;
	std::cout << "neww: " << neww << std::endl;

	cv::resize(image, image, cv::Size(neww, newh));

	std::cout << "image.cols: " << image.cols << std::endl;
	std::cout << "image.rows: " << image.rows << std::endl;
}
