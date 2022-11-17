#include <opencv2/imgcodecs/imgcodecs.hpp>
#include <torch/torch.h>
#include <torch/script.h>
#include <iostream>
#include <memory>
#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

// https://github.com/AllentDan/LibtorchSegmentation/blob/main/src/Segmentor.h
int main(){

	std::cout << "The current OpenCV version is " << CV_VERSION << "\n";
	torch::Tensor tensor = torch::rand({2, 3});
	std::cout << tensor << std::endl;

	// see http://host.robots.ox.ac.uk:8080/pascal/VOC/voc2007/segexamples/index.html for the list of classes with indexes
	const int CLASSNUM = 21;
	const int DOG = 12;
	const int PERSON = 15;
	const int SHEEP = 17;

	const int inputSize = 224;
	const float TORCHVISION_NORM_MEAN_RGB[] = {0.485, 0.456, 0.406};
	const float TORCHVISION_NORM_STD_RGB[] = {0.229, 0.224, 0.225};


	torch::jit::script::Module module;

	try {
		module = torch::jit::load("/home/cyrill/dev/models/deeplabv3.pt");
	}
	catch (const c10::Error& e) {
		std::cerr << "error loading the model\n";
		return -1;
	}


	std::cout << "Successfully loaded\n";
	cv::Mat jpg;
	jpg = cv::imread("/home/cyrill/Pictures/deeplab.jpg");
	if (jpg.empty())
	{
		std::cout << "!!! Failed imread(): image not found" << std::endl;
		// don't let the execution continue, else imshow() will crash.
	}

	std::cout << "Seems to have worked" << std::endl;
	cv::Mat resized;
	try {
		cv::resize(jpg, resized, cv::Size(inputSize, inputSize));
	} catch (const std::exception& e) {
		std::cout << e.what();
	}

	// todo: normalize with mean, std
}
