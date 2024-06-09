//
// Created by cyrill on 16.11.23.
//

#ifndef EXAMPLE_APP_PREDICTOR_H
#define EXAMPLE_APP_PREDICTOR_H

#ifndef SAM_PREDICTOR_H
#define SAM_PREDICTOR_H


#include <torch/script.h>
#include <torch/torch.h>
#include "resize_longest_size.h"

class SamPredictor {
    public:
	SamPredictor(int target_length, const std::string &predictor_model_path,
		     const std::string &image_embedding_model_path)
		: transform(target_length)
		, isImageSet(false)
		, originalImageWidth(0)
		, originalImageHeight(0)
	{
		resetImage();
		predictorModel = torch::jit::load(predictor_model_path);
		imageEmbeddingModel =
			torch::jit::load(image_embedding_model_path);
	}

	void setImage(const cv::Mat &image);
	void resetImage();

	std::tuple<torch::Tensor, torch::Tensor, torch::Tensor>
	predict(const torch::Tensor &pointCoordsVec,
		const torch::Tensor &pointLabelsVec,
		const torch::Tensor &maskInput, bool maskInputBool);
	int originalImageWidth;
	int originalImageHeight;
	ResizeLongestSide transform;

    private:
	torch::jit::script::Module predictorModel;
	torch::jit::script::Module imageEmbeddingModel;
	torch::Tensor features;
	bool isImageSet;

	void preProcess(torch::Tensor &inputTensor);

	std::vector<float> pixelMean = { 123.675, 116.28, 103.53 };
	std::vector<float> pixelStd = { 58.395, 57.12, 57.375 };
	std::pair<int, int> inputSize = { 1024, 1024 };

	void setTorchImage(torch::Tensor &inputTensor);

	void debugPrint(const c10::ivalue::TupleElements &outputs) const;
};

#endif // SAM_PREDICTOR_H
#endif //EXAMPLE_APP_PREDICTOR_H
