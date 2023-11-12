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



std::tuple<int, int> get_preprocess_shape(int oldh, int oldw, int long_side_length) {
    double scale = static_cast<double>(long_side_length) / std::max(oldh, oldw);
    int newh = static_cast<int>(oldh * scale + 0.5);
    int neww = static_cast<int>(oldw * scale + 0.5);
    return {newh, neww};
}

torch::Tensor preProcess(cv::Mat& img, int inputWidth, int inputHeight) {

// img is  8-bit unsigned integer format! 

    // Resize the image
    cv::resize(img, img, cv::Size(inputWidth, inputHeight));

    // Convert to tensor
    auto inputTensor = torch::from_blob(
        img.data, {img.rows, img.cols, 3}, torch::kByte);
    
	std::cout << "Created a tensfor from th image (which was resized before)" << std::endl;

    // like `set_image?  form SAMPredictor(
    //
    inputTensor = inputTensor.permute({2, 0, 1}).contiguous();
	inputTensor = inputTensor.unsqueeze(0); 

                                             
	std::cout << "permute ok" << std::endl;
	if (!(inputTensor.sizes().size() == 4 && inputTensor.size(1) == 3)) {
		throw std::runtime_error(
			"set_torch_image input must be BCHW with long side");
	}
    std::cout << "assert ok" << std::endl;

    const std::vector<float> pixel_mean = { 123.675, 116.28, 103.53 };
    const std::vector<float> pixel_std = { 58.395, 57.12, 57.375 };
    torch::Tensor tensor_pixel_mean = torch::tensor(pixel_mean);
    torch::Tensor tensor_pixel_std = torch::tensor(pixel_std);

    // Reshape mean and std tensors for broadcasting
    tensor_pixel_mean = tensor_pixel_mean.view({1, 3, 1, 1});
    tensor_pixel_std = tensor_pixel_std.view({1, 3, 1, 1});

    std::cout << "create tensors ok" << std::endl;

    // this fails
    inputTensor = (inputTensor - tensor_pixel_mean) / tensor_pixel_std;

    
	std::cout << "norm ok" << std::endl;

    // Pad
    int padh = img.rows - inputWidth;
    int padw = img.cols - inputHeight; 
    torch::nn::functional::PadFuncOptions padOptions({0, padw, 0, padh});
    inputTensor = torch::nn::functional::pad(inputTensor, padOptions);
        
	std::cout << "pad ok" << std::endl;


    return inputTensor;
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

	cv::Mat jpg;
	jpg = cv::imread("/home/cyrill/Downloads/img.jpg");

	if (jpg.empty()) {
		std::cout << "!!! Failed imread(): image not found"
			  << std::endl;
		return -1;
	}

	cv::Mat img_converted;
	cv::cvtColor(jpg, img_converted, cv::COLOR_BGR2RGB);


	cv::Mat img = img_converted;

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

    // now we hvae features. So now we are at a state where we have called
    // predictor = SAMPredictor()
    // predictor.set_image()


	std::cout << ".forward called on imageEmbeddingModel" << std::endl;
	std::cout << "we have features" << std::endl;

    // what we do now tries to be equivalent to the following python from the
    // example: 
    // masks, scores, logits = predictor.predict(
    //point_coords=input_point,
    //point_labels=input_label,
    //kmultimask_output=True,
//)


	// Prepare additional inputs for the predictor model
	std::vector<int64_t> pointCoords = { 20, 20 }; // random point for now
	std::vector<int64_t> pointLabels = { 1 }; // Example label
                                              
	
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
