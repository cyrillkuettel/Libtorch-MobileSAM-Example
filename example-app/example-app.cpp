#include <iostream>
#include <memory>
#include <vector>
#include <stdexcept>
#include <string>
#include <sstream>
#include <stdlib.h>

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgcodecs/imgcodecs.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/opencv.hpp>

#include <torch/script.h>
#include <torch/torch.h>

const int inputWidth = 1024;
const int inputHeight = 1024;

int originalImageHeight;
int originalImageWidth;

const std::vector<float> pixel_mean = { 123.675, 116.28, 103.53 };
const std::vector<float> pixel_std = { 58.395, 57.12, 57.375 };

// This command line application with three (optional) arguments:
// 1. <image_path>: The path to the image file you want to process (e.g., "img.jpg").
// 2. <mobilesam_predictor_model_path>: The path to the MobileSAM predictor model file
// 3. <vit_image_embedding_model_path>: The path to the image embedding model file

// usage: ./example-app img.jpg mobilesam_preditor.pt vit_image_embedding.pt

struct ImageParams {
	std::string image;
	std::string mobilesamPredictor;
	std::string vitImageEmbedding;
};

ImageParams parseParams(int argc, char *argv[])
{
	ImageParams params;

	// adjust what you want, or use the input arguments:

	std::string defaultImagePath =
		"/home/cyrill/pytorch/libtorch-opencv/example-app/images/picture1.jpg";
	std::string defaultMobileSamPredictor =
		"/home/cyrill/pytorch/libtorch-opencv/example-app/models/mobilesam_predictor.pt";
	std::string defaultVitImageEmbedding =
		"/home/cyrill/pytorch/libtorch-opencv/example-app/models/vit_image_embedding.pt";

	params.image = (argc > 1) ? argv[1] : defaultImagePath;
	params.mobilesamPredictor = (argc > 2) ? argv[2] :
						 defaultMobileSamPredictor;
	params.vitImageEmbedding = (argc > 3) ? argv[3] :
						defaultVitImageEmbedding;
	return params;
}

torch::Tensor preProcess(cv::Mat &img, int inputWidth, int inputHeight)
{
	// note: img is 8-bit unsigned integer format!

	cv::resize(img, img, cv::Size(inputWidth, inputHeight));

	auto inputTensor = torch::from_blob(img.data, { img.rows, img.cols, 3 },
					    torch::kByte);

	// kind of like `set_image` from SAMPredictor

	inputTensor = inputTensor.permute({ 2, 0, 1 }).contiguous();
	inputTensor = inputTensor.unsqueeze(0);

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

void showPoints(const torch::Tensor &coords, const torch::Tensor &labels,
		cv::Mat &image, int markerSize = 6)
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
		      const torch::Tensor &pointCoords,
		      const torch::Tensor &pointLabels)
{
	std::cout << "Visualizing " << masks.size(0) << " masks.\n";

	// Loop through each mask
	for (int i = 0; i < masks.size(0); i++) {
		cv::Mat displayImage = image.clone();
		showMask(masks[i], displayImage);
		showPoints(pointCoords, pointLabels, displayImage);
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

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor>
runInference(torch::Tensor inputTensor,
	     torch::jit::script::Module &imageEmbeddingModel,
	     torch::jit::script::Module &predictorModel,
	     torch::Tensor &pointCoordsTensor, torch::Tensor &pointLabelsTensor,
	     int originalImageHeight, int originalImageWidth)
{
	// up until here, this was the `SamPredictor.set_image` function which does
	// all the pre-processing

	// https://github.com/cmarschner/MobileSAM/blob/a509aac54fdd7af59f843135f2f7cee307283c88/mobile_sam/predictor.py#L79
	//

	// Run image through the embedding model
	// Here we differ from the orignial python example because we have two
	// models for pytroch lite interpreter
	std::vector<torch::jit::IValue> inputs;
	inputs.emplace_back(inputTensor);

	// image embeddings:
	auto features = imageEmbeddingModel.forward(inputs).toTensor();

	// now we hvae features. So now basically we are at a state that is similar to the python state of where we have called
	// predictor = SAMPredictor()
	// predictor.set_image()

	std::cout << ".forward called on imageEmbeddingModel" << std::endl;

	// now (equivalent python) this:
	// masks, scores, logits = predictor.predict(
	//point_coords=input_point,
	//point_labels=input_label,
	//kmultimask_output=True,
	//)

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
	inputs2.emplace_back(features); // image_embeddings
	inputs2.emplace_back(pointCoordsTensor);
	inputs2.emplace_back(pointLabelsTensor);
	inputs2.emplace_back(maskInput);
	inputs2.emplace_back(hasMaskInput);
	inputs2.emplace_back(origImgSize);

	auto modeloutput = predictorModel.forward(inputs2);
	// Accessing output tensors

	auto outputs = modeloutput.toTuple()->elements();

	std::cout << "output tensors informations:" << std::endl;

	for (size_t i = 0; i < outputs.size(); ++i) {
		auto tensor = outputs[i].toTensor();
		std::string variable_name;
		switch (i) {
		case 0:
			variable_name = "masks";
			break;
		case 1:
			variable_name = "iou_predictions";
			break;
		case 2:
			variable_name = "low_res_masks";
			break;
		default:
			variable_name = "Output " + std::to_string(i);
		}
		std::cout << variable_name << ": Size = " << tensor.sizes()
			  << ", Type = " << tensor.scalar_type() << std::endl;
	}
	torch::Tensor masks = outputs[0].toTensor();
	torch::Tensor iou_predictions = outputs[1].toTensor();
	torch::Tensor low_res_masks = outputs[2].toTensor();

	return std::make_tuple(masks, iou_predictions, low_res_masks);
}

int main(int argc, char *argv[])

{
	std::cout << "The current OpenCV version is " << CV_VERSION << "\n";
	std::cout << "LibTorch version: " << TORCH_VERSION << std::endl;

	ImageParams params;
	try {
		params = parseParams(argc, argv);
	} catch (const std::runtime_error &e) {
		std::cerr << e.what() << "\n";
		return -1;
	}

	std::cout << "OpenCV version is " << CV_VERSION << "\n";

	torch::jit::script::Module predictorModel;
	torch::jit::script::Module imageEmbeddingModel;

	std::string mobilesam_predictor = params.mobilesamPredictor;
	std::string vit_image_embedding = params.vitImageEmbedding;
	std::cout << "MobileSAM Predictor Path: " << params.mobilesamPredictor
		  << "\n";
	std::cout << "ViT Image Embedding Path: " << params.vitImageEmbedding
		  << "\n";

	try {
		predictorModel = torch::jit::load(mobilesam_predictor);
		imageEmbeddingModel = torch::jit::load(vit_image_embedding);
	} catch (const c10::Error &e) {
		std::cerr << e.what() << "error loading the model\n";
		return -1;
	}

	cv::Mat jpg = cv::imread(params.image, cv::IMREAD_COLOR);

	if (jpg.empty()) {
		std::cout << "imread(): failed: Image not found" << std::endl;
		return -1;
	}

	cv::Mat img_converted;
	cv::cvtColor(jpg.clone(), img_converted, cv::COLOR_BGR2RGB);
	cv::Mat img = img_converted;

	originalImageHeight = img.rows;
	originalImageWidth = img.cols;

	// img_converted.convertTo(img, CV_8UC3) seems redundant in this context because the image, once loaded and potentially color-converted, is already in the CV_8UC3 format. But it pays to be paranoid...
	img_converted.convertTo(img, CV_8UC3);

	torch::Tensor inputTensor = preProcess(img, inputWidth, inputHeight);

	// define some input_points for the model:
//	std::vector<float> pointCoords = { 371.0f, 190.0f, 374.0f, 190.0f, 365.0f, 190.0f, 371.0f, 190.0f, 371.0f, 190.0f };

    std::vector<float> pointCoords(10, 0.0f);
	pointCoords[0] = 20.0f;
	pointCoords[1] = 20.0f;
	std::vector<float> pointLabels = { 1.0f, 0.0f, 0.0f, 0.0f, 0.0f };

	auto pointCoordsTensor =
		torch::tensor(pointCoords, torch::dtype(torch::kFloat32));

	auto pointLabelsTensor =
		torch::tensor(pointLabels, torch::dtype(torch::kFloat32));

	pointCoordsTensor =
		pointCoordsTensor.reshape({ 1, 5, 2 }).to(torch::kFloat32);
	pointLabelsTensor =
		pointLabelsTensor.reshape({ 1, 5 }).to(torch::kFloat32);

	// 'iou_predictions': The model's predictions of mask quality, in shape BxC.
	// 'masks': Batched binary mask predictions with shape BxCxHxW,

	torch::Tensor masks, iou_predictions, low_res_masks;
	std::tie(masks, iou_predictions, low_res_masks) =
		runInference(inputTensor, imageEmbeddingModel, predictorModel,
			     pointCoordsTensor, pointLabelsTensor,
			     originalImageHeight, originalImageWidth);

	std::cout << masks.sizes() << std::endl;

	visualizeResults(jpg, masks, iou_predictions, pointCoordsTensor,
			 pointLabelsTensor);

	return 0;
}
