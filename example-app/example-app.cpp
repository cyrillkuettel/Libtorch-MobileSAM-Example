#include <iostream>
#include <memory>
#include <vector>
#include <cassert>
#include <stdexcept>
#include <algorithm>

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

torch::Tensor preProcess(cv::Mat &img, int inputWidth, int inputHeight)
{
	// note: img is 8-bit unsigned integer format!

	cv::resize(img, img, cv::Size(inputWidth, inputHeight));

	auto inputTensor = torch::from_blob(img.data, { img.rows, img.cols, 3 },
					    torch::kByte);

	// kind of like `set_image` from SAMPredictor

	inputTensor = inputTensor.permute({ 2, 0, 1 }).contiguous();
	inputTensor = inputTensor.unsqueeze(0);

	std::cout << "permute ok" << std::endl;
	if (!(inputTensor.sizes().size() == 4 && inputTensor.size(1) == 3)) {
		throw std::runtime_error(
			"set_torch_image input must be BCHW with long side");
	}

	torch::Tensor tensor_pixel_mean = torch::tensor(pixel_mean);
	torch::Tensor tensor_pixel_std = torch::tensor(pixel_std);

	// Reshape mean and std tensors for broadcasting
	tensor_pixel_mean = tensor_pixel_mean.view({ 1, 3, 1, 1 });
	tensor_pixel_std = tensor_pixel_std.view({ 1, 3, 1, 1 });
	inputTensor = (inputTensor - tensor_pixel_mean) / tensor_pixel_std;

	// Pad
	int padh = img.rows - inputWidth;
	int padw = img.cols - inputHeight;
	torch::nn::functional::PadFuncOptions padOptions({ 0, padw, 0, padh });
	inputTensor = torch::nn::functional::pad(inputTensor, padOptions);
	return inputTensor;
}

int main()
{
	std::cout << "OpenCV version is " << CV_VERSION << "\n";

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

	cv::Mat jpg;
	jpg = cv::imread("/home/cyrill/Downloads/img.jpg");

	if (jpg.empty()) {
		std::cout << "imread(): failed: Image not found" << std::endl;
		return -1;
	}

	cv::Mat img_converted;
	cv::cvtColor(jpg, img_converted, cv::COLOR_BGR2RGB);

	cv::Mat img = img_converted;
	int originalImageHeight = img.rows;
	int originalImageWidth = img.cols;

	std::cout << "convertTo" << std::endl;
	// The line img_converted.convertTo(img, CV_8UC3); seems redundant in this context because the image, once loaded and potentially color-converted, is already in the CV_8UC3 format.
	//
	img_converted.convertTo(img, CV_8UC3);

	// resize accoring to how they do it
	// not rocket science
	torch::Tensor inputTensor = preProcess(img, inputWidth, inputHeight);

	// up until here, this was the `SamPredictor.set_image` function which does
	// all the pre-processing

	std::cout << "preProcess ok" << std::endl;

	// https://github.com/cmarschner/MobileSAM/blob/a509aac54fdd7af59f843135f2f7cee307283c88/mobile_sam/predictor.py#L79
	//

	// Run image through the embedding model
	// Here we differ from the orignial python example because we have two
	// models for pytroch lite interpreter
	std::vector<torch::jit::IValue> inputs;
	inputs.push_back(inputTensor);

	// image embeddings:
	auto features = imageEmbeddingModel.forward(inputs).toTensor();

	// now we hvae features. So now basically we are at a state that is similar to the python state of where we have called
	// predictor = SAMPredictor()
	// predictor.set_image()

	std::cout << ".forward called on imageEmbeddingModel" << std::endl;
	std::cout << "we have features" << std::endl;

	// now (equivalent python) this:
	// masks, scores, logits = predictor.predict(
	//point_coords=input_point,
	//point_labels=input_label,
	//kmultimask_output=True,
	//)

	// todo: replace with actual touched point when we have that.
	std::vector<float> pointCoords = {
		20.0f, 20.0f, 20.0f, 20.0f, 20.0f,
		20.0f, 20.0f, 20.0f, 20.0f, 20.0f
	}; // Random points as floats
	std::vector<float> pointLabels = { 1.0f, 1.0f, 1.0f, 1.0f,
					   1.0f }; // Example labels as floats

	auto pointCoordsTensor =
		torch::tensor(pointCoords, torch::dtype(torch::kFloat32));
	auto pointLabelsTensor =
		torch::tensor(pointLabels, torch::dtype(torch::kFloat32));

	pointCoordsTensor =
		pointCoordsTensor.reshape({ 1, 5, 2 }).to(torch::kFloat32);
	pointLabelsTensor =
		pointLabelsTensor.reshape({ 1, 5 }).to(torch::kFloat32);

	/**
     * predictorModel.forward(
        Tensor image_embeddings, 
        Tensor point_coords,
        Tensor point_labels, 
        Tensor mask_input, 
        Tensor has_mask_input, 
        Tensor orig_im_size)
    */

	// int mask_input_size = [4 * x for x in embed_size] == [256, 256]

	auto maskInput =
		torch::zeros({ 1, 1, 256, 256 }, torch::dtype(torch::kFloat32));

	// origImgSize might have to be [1500, 2250] I'm not sure at this point
	auto origImgSize =
		torch::tensor({ originalImageHeight, originalImageWidth },
			      torch::dtype(torch::kFloat32));

	auto hasMaskInput = torch::tensor({ 0 }, torch::dtype(torch::kFloat32));
	std::vector<torch::jit::IValue> inputs2;
	inputs2.push_back(features); // image_embeddings
	inputs2.push_back(pointCoordsTensor);
	inputs2.push_back(pointLabelsTensor);
	inputs2.push_back(maskInput);
	inputs2.push_back(hasMaskInput);
	inputs2.push_back(origImgSize);

	auto modeloutput = predictorModel.forward(inputs2);
	std::cout << "predictorModel.forward(inputs2); ok " << std::endl;
	// Accessing output tensors

	auto outputs = modeloutput.toTuple()->elements();

	for (size_t i = 0; i < outputs.size(); ++i) {
		auto tensor = outputs[i].toTensor();
		std::cout << "Output " << i << ": Size = " << tensor.sizes()
			  << ", Type = " << tensor.scalar_type() << std::endl;
	}
	torch::Tensor masks = outputs[0].toTensor();
	torch::Tensor iou_predictions = outputs[1].toTensor();
	torch::Tensor low_res_masks = outputs[2].toTensor();

	// Process the output as needed
}
