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

class SamPredictor {
    public:
	SamPredictor(const torch::jit::script::Module &embeddingModel,
		     const torch::jit::script::Module &predictorModel)
		: embeddingModel(embeddingModel)
		, predictorModel(predictorModel)
		, isImageSet(false)
	{
	}

	void set_image(cv::Mat &image)
	{
		cv::Mat resizedImage;
		cv::resize(
			image, resizedImage,
			cv::Size(1024,
				 1024)); // Resize to expected input size for embedding model

		torch::Tensor imageTensor =
			torch::from_blob(resizedImage.data,
					 { resizedImage.rows, resizedImage.cols,
					   3 },
					 torch::kByte)
				.clone();
		imageTensor = imageTensor.to(torch::kFloat)
				      .permute({ 2, 0, 1 })
				      .unsqueeze(0);

		torch::Tensor embeddingOutput =
			embeddingModel.forward({ imageTensor }).toTensor();

		this->features = embeddingOutput;
		this->isImageSet = true;
	}

	std::tuple<torch::Tensor, torch::Tensor, torch::Tensor>
	predict(const torch::Tensor &pointCoords,
		const torch::Tensor &pointLabels,
		const torch::Tensor &maskInput = torch::Tensor(),
		const bool multimaskOutput = true,
		const bool returnLogits = false)
	{
		if (!isImageSet) {
			throw std::runtime_error(
				"An image must be set before mask prediction.");
		}

		// Prepare inputs for the predictor model
		torch::Tensor hasMaskInput = torch::tensor(
			{ maskInput.defined() ? 1 : 0 }, torch::kFloat);
		torch::Tensor origImSize = torch::tensor(
			{ 453, 680}, // original Image size
                        
			torch::kFloat); // Update with actual image size

		std::vector<torch::jit::IValue> inputs = {
			features,
			pointCoords.unsqueeze(0),
			pointLabels.unsqueeze(0),
			maskInput.defined() ? maskInput.unsqueeze(0) :
					      torch::Tensor(),
			hasMaskInput,
			origImSize
		};

		// Predict masks
		auto output =
			predictorModel.forward(inputs).toTuple()->elements();
		torch::Tensor masks = output[0].toTensor();
		torch::Tensor iouPredictions = output[1].toTensor();
		torch::Tensor lowResMasks = output[2].toTensor();

		if (!returnLogits) {
			masks = masks >
				0.5; // Apply threshold to convert logits to binary mask
		}

		return std::make_tuple(masks, iouPredictions, lowResMasks);
	}

    private:
	torch::jit::script::Module embeddingModel;
	torch::jit::script::Module predictorModel;
	torch::Tensor features;
	bool isImageSet;
};

// Example usage
int main()
{
	torch::jit::script::Module embeddingModel = torch::jit::load(
		"/home/cyrill/Documents/models/vit_image_embedding.pt");
	torch::jit::script::Module predictorModel = torch::jit::load(
		"/home/cyrill/Documents/models/mobilesam_predictor.pt");

	SamPredictor predictor(embeddingModel, predictorModel);

	cv::Mat jpg = cv::imread("/home/cyrill/Downloads/img.jpg");
	if (jpg.empty()) {
		std::cout << "Failed imread(): image not found" << std::endl;
		return -1;
	}

	predictor.set_image(jpg);

	// Prepare prediction inputs
    // create hardcoded input points for now. 
    
	torch::Tensor pointCoords = torch::tensor({ 400, 400 }, torch::kInt64)

					    .to(torch::kFloat32)
					    .reshape({ 1, 2 });
	torch::Tensor pointLabels = torch::tensor({ 1 }, torch::kInt32);

	// should be [masks, iouPredictions, lowResMasks]

	auto predictionResult = predictor.predict(pointCoords, pointLabels);
    auto masks = std::get<0>(predictionResult);
    auto iouPredictions = std::get<1>(predictionResult);
    auto lowResMasks = std::get<2>(predictionResult);

	// Process the outputs (e.g., display or save the masks)
	// ...

	return 0;
}
