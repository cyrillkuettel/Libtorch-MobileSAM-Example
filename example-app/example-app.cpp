#include <iostream>
#include <memory>
#include <vector>

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgcodecs/imgcodecs.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/opencv.hpp>

#include <torch/script.h>
#include <torch/torch.h>


	const int inputSize = 1024;

	const int inputWidth = 1024;
	const int inputHeight = 1024;

	const std::vector<float> pixel_mean = { 123.675, 116.28, 103.53 };
	const std::vector<float> pixel_std = { 58.395, 57.12, 57.375 };

torch::Tensor preprocess(const torch::Tensor &input, int targetSize)
{
	// Normalize colors
	auto meanTensor = torch::tensor(pixel_mean).view({ -1, 1, 1 });
	auto stdTensor = torch::tensor(pixel_std).view({ -1, 1, 1 });
	auto normalizedInput = (input - meanTensor) / stdTensor;

	// Pad to square
	int h = input.size(1);
	int w = input.size(2);
	int padh = targetSize - h;
	int padw = targetSize - w;
	auto paddedInput = torch::nn::functional::pad(
		normalizedInput,
		torch::nn::functional::PadFuncOptions({ 0, padw, 0, padh }));

	return paddedInput;
}

int main()
{
	std::cout << "The current OpenCV version is " << CV_VERSION << "\n";
	torch::Tensor tensor = torch::rand({ 2, 3 });
	std::cout << tensor << std::endl;


	torch::jit::script::Module predictorModel;
	torch::jit::script::Module imageEmbeddingModel;

	std::string mobilesam_predictor =
		"/home/cyrill/Documents/models/mobilesam_predictor.pt";

	std::string vit_image_embedding =
		"/home/cyrill/Documents/models/vit_image_embedding.pt";

	try {
		predictorModel = torch::jit::load(mobilesam_predictor);
		imageEmbeddingModel = torch::jit::load(vit_image_embedding);

	} catch (const c10::Error &e) {
		std::cerr << e.what() << "error loading the model\n";
		return -1;
	}

	std::cout << "Successfully loaded\n";
	cv::Mat jpg;
	jpg = cv::imread("/home/cyrill/Downloads/img.jpg");

	if (jpg.empty()) {
		std::cout << "!!! Failed imread(): image not found"
			  << std::endl;
        return -1;
	
	}

    cv::Mat img_converted;
    cv::cvtColor(jpg, img_converted, cv::COLOR_BGR2RGB);

    cv::Mat img;
    img.convertTo(img, CV_8UC3); // Converts the image to uint8 with 3 channels (HxWxC)
    
    cv::resize(img, img, cv::Size(inputWidth, inputHeight));
    

	// like `set_image` of `SamPredictor`
	auto inputTensor = torch::from_blob(img.data, { 1, 3, inputWidth, inputHeight }, torch::kFloat);
    inputTensor = inputTensor.permute({2, 0, 1}).contiguous(); // Change layout from HWC to CHW
    inputTensor = inputTensor.unsqueeze(0); // Add batch dimension
	std::cout << "from_blob ok" << std::endl;


	// Run image through the embedding model
    std::vector<torch::jit::IValue> inputs;
    inputs.push_back(inputTensor);

	auto features = imageEmbeddingModel.forward(inputs).toTensor();

	std::cout << ".forward called" << std::endl;


	// Prepare additional inputs for the predictor model
	std::vector<int64_t> pointCoords = { 200, 200 }; // random point for now
	std::vector<int64_t> pointLabels = { 1 }; // Example label
                                              //
	auto pointCoordsTensor =
		torch::tensor(pointCoords, torch::dtype(torch::kInt64));
	auto pointLabelsTensor =
		torch::tensor(pointLabels, torch::dtype(torch::kInt64));
	pointCoordsTensor =
		pointCoordsTensor.reshape({ 1, 1, 2 }).to(torch::kFloat32);
	pointLabelsTensor =
		pointLabelsTensor.reshape({ 1, 1 }).to(torch::kInt32);

	// Run the prediction model
	inputs = { features, pointCoordsTensor,
		   pointLabelsTensor }; // Add more inputs as needed
	auto outputDict = predictorModel.forward(inputs).toGenericDict();

	// Accessing output tensors
	auto masks = outputDict.at("masks").toTensor();
	auto iou_predictions = outputDict.at("iou_predictions").toTensor();
	auto low_res_masks = outputDict.at("low_res_logits").toTensor();

	// Process the output as needed
	// ...
}


