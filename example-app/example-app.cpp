#include <iostream>
#include <memory>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgcodecs/imgcodecs.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/opencv.hpp>
#include <torch/script.h>
#include <torch/torch.h>


void test()
{
	const unsigned char colors[37][3] = {
		{ 54, 67, 244 },   { 99, 30, 233 },   { 176, 39, 156 },
		{ 183, 58, 103 },  { 181, 81, 63 },   { 243, 150, 33 },
		{ 244, 169, 3 },   { 212, 188, 0 },   { 136, 150, 0 },
		{ 80, 175, 76 },   { 74, 195, 139 },  { 57, 220, 205 },
		{ 59, 235, 255 },  { 7, 193, 255 },   { 0, 152, 255 },
		{ 34, 87, 255 },   { 72, 85, 121 },   { 158, 158, 158 },
		{ 139, 125, 96 },  { 124, 32, 36 },   { 40, 200, 40 },
		{ 32, 32, 200 },   { 231, 129, 255 }, { 32, 10, 34 },
		{ 124, 34, 120 },  { 120, 32, 200 },  { 31, 129, 255 },
		{ 132, 10, 34 },   { 124, 100, 220 }, { 155, 255, 180 },
		{ 217, 220, 105 }, { 19, 35, 155 },   { 100, 193, 255 },
		{ 0, 152, 55 },	   { 25, 125, 25 },   { 122, 122, 255 },
		{ 0, 120, 87 }
	};

	const unsigned char* color = colors[10];
	const unsigned char mycolor1 = color[0];
	const unsigned char mycolor2 = color[1];

	std::cout << static_cast<unsigned>(mycolor1) << std::endl;
	std::cout << static_cast<unsigned>(mycolor2) << std::endl;
}

// https://github.com/AllentDan/LibtorchSegmentation/blob/main/src/Segmentor.h
// https://docs.opencv.org/4.3.0/d4/d88/samples_2dnn_2segmentation_8cpp-example.html#a19
int main()
{


	std::cout << "The current OpenCV version is " << CV_VERSION << "\n";
	torch::Tensor tensor = torch::rand({ 2, 3 });
	std::cout << tensor << std::endl;

	// see
	// http://host.robots.ox.ac.uk:8080/pascal/VOC/voc2007/segexamples/index.html
	// for the list of classes with indexes
	const int CLASSNUM = 21;
	const int DOG = 12;
	const int PERSON = 15;
	const int SHEEP = 17;

	const int inputSize = 224;
	const float TORCHVISION_NORM_MEAN_RGB[] = { 0.485, 0.456, 0.406 };
	const float TORCHVISION_NORM_STD_RGB[] = { 0.229, 0.224, 0.225 };

	torch::jit::script::Module module;

	try {
		module = torch::jit::load(
			"/home/cyrill/dev/models/deeplabv3.pt");
	} catch (const c10::Error &e) {
		std::cerr << "error loading the model\n";
		return -1;
	}

	std::cout << "Successfully loaded\n";
	cv::Mat jpg;
	jpg = cv::imread("/home/cyrill/pytorch/example-app/deeplab.jpg");
	if (jpg.empty()) {
		std::cout << "!!! Failed imread(): image not found"
			  << std::endl;
		// don't let the execution continue, else imshow() will crash.
	}

	std::cout << "Seems to have worked" << std::endl;
	cv::Mat resized;
	try {
		cv::resize(jpg, resized, cv::Size(inputSize, inputSize));
	} catch (const std::exception &e) {
		std::cout << e.what();
	}

	// todo: normalize with mean, std
	// https://github.com/pytorch/pytorch/issues/14273

}
