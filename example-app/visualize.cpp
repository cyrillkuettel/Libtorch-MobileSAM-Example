#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <sstream>
#include <cstdlib>

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgcodecs/imgcodecs.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/opencv.hpp>

#include <torch/script.h>

cv::Mat tensorToMat(torch::Tensor tensor)
{
	tensor = tensor.squeeze().detach();
	if (tensor.dtype() != torch::kU8) {
		tensor = tensor.to(torch::kU8);
	}

	cv::Mat mat(tensor.size(0), tensor.size(1), CV_8UC1,
		    tensor.data_ptr<uchar>());
	return mat.clone(); // Clone to safely detach from the tensor data
}

void showMask(const torch::Tensor &mask, cv::Mat &image)
{
	if (mask.numel() == 0) {
		throw std::runtime_error("Empty mask tensor.");
	}

	cv::Scalar color = cv::Scalar(30, 144, 255); // BGR color
	double alpha = 0.3; // Transparency factor

	cv::Mat maskMat = tensorToMat(mask);
	cv::Mat coloredMask;
	cv::cvtColor(maskMat, coloredMask, cv::COLOR_GRAY2BGR);

	std::vector<cv::Mat> channels(3);
	cv::split(coloredMask, channels);
	channels[0] *= color[0];
	channels[1] *= color[1];
	channels[2] *= color[2];
	cv::merge(channels, coloredMask);

	// Overlay the colored mask on the image
	cv::addWeighted(image, 1.0 - alpha, coloredMask, alpha, 0.0, image);
}

void showPoints(const torch::Tensor &coords, cv::Mat &image, int markerSize = 6)
{
	if (coords.sizes().size() != 3 || coords.size(1) != 5 ||
	    coords.size(2) != 2) {
		throw std::runtime_error(
			"coords must be a 3D tensor with shape [1, N, 2].");
	}

	auto batch = coords[0];
	std::cout << "Showing the " << batch.size(0) << " points.\n";
	for (int i = 0; i < batch.size(0); ++i) {
		int x = batch[i][0].item<int>();
		int y = batch[i][1].item<int>();
		cv::circle(image, cv::Point(x, y), markerSize,
			   cv::Scalar(0, 255, 0), -1); // Green color for points
	}
}

// Main visualization function
void visualizeResults(cv::Mat &image, const torch::Tensor &masks,
		      const torch::Tensor &scores,
		      const torch::Tensor &pointCoords)
{
	std::cout << "Visualizing " << masks.size(0) << " masks.\n";

	// Loop through each mask
	for (int i = 0; i < masks.size(0); i++) {
		cv::Mat displayImage = image.clone();
		showMask(masks[i], displayImage);
		showPoints(pointCoords, displayImage);
		// Show box if needed

		std::time_t t = std::time(nullptr);
		std::tm tm = *std::localtime(&t);
		std::ostringstream oss;
		oss << std::put_time(&tm, "%Y%m%d_%H%M%S");
		std::string timestamp = oss.str();

		std::string filename = "MobileSAM" + std::to_string(i + 1) +
				       "_" + timestamp + ".jpg";

		cv::imwrite(filename, displayImage);

		char actualpath[PATH_MAX + 1];
		char *ptr;
		ptr = realpath(filename.c_str(), actualpath);

		if (ptr) {
			std::cout << "Saved output image:" << ptr << std::endl;
		} else {
			std::cout << "Saved output image: " << filename
				  << std::endl;
		}

		cv::imshow("Visualization - Mask " + std::to_string(i + 1) +
				   ", Score: " +
				   std::to_string(scores[i].item<float>()),
			   displayImage);
		cv::waitKey(0); // Wait for a key press
	}
}
