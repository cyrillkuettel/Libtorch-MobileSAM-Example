#include <opencv2/imgcodecs/imgcodecs.hpp>
#include <torch/torch.h>
#include <torch/script.h>
#include <iostream>
#include <memory>
#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

int main(){

	std::cout << "The current OpenCV version is " << CV_VERSION << "\n";
	torch::Tensor tensor = torch::rand({2, 3});
	std::cout << tensor << std::endl;

	// see http://host.robots.ox.ac.uk:8080/pascal/VOC/voc2007/segexamples/index.html for the list of classes with indexes
	const int CLASSNUM = 21;
	const int DOG = 12;
	const int PERSON = 15;
	const int SHEEP = 17;

	torch::jit::script::Module module;

	try {
		// Deserialize the ScriptModule from a file using torch::jit::load().
		module = torch::jit::load("/home/cyrill/dev/models/deeplabv3.pt");
	}
	catch (const c10::Error& e) {
		std::cerr << "error loading the model\n";
		return -1;
	}


	std::cout << "Successfully loaded\n";
	cv::Mat jpg;
	jpg = cv::imread("/home/cyrill/Documents/faces/unsplash.jpg");
	if (jpg.empty())
	{
		std::cout << "!!! Failed imread(): image not found" << std::endl;
		// don't let the execution continue, else imshow() will crash.
	}


}
