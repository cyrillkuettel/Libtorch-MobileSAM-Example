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

	std::pair<int, int> getPreprocessShape(int oldh, int oldw) const;
	void applyImage(cv::Mat &image);

	std::vector<std::pair<float, float> >
	applyCoords(const std::vector<std::pair<float, float> > &coords,
		    const std::pair<int, int> &original_size) const;

    private:
	int target_length;
};

#endif //EXAMPLE_APP_TRANSFORMS_H

#endif //EXAMPLE_APP_RESIZE_LONGEST_SIZE_H
