#include "predictor.h"
#include "visualize.cpp"

int main()
{
	std::string defaultImagePath =
		"/home/cyrill/pytorch/libtorch-opencv/example-app/images/img.jpg";
	std::string defaultMobileSamPredictor =
		"/home/cyrill/pytorch/libtorch-opencv/example-app/models/mobilesam_predictor.pt";
	std::string defaultVitImageEmbedding =
		"/home/cyrill/pytorch/libtorch-opencv/example-app/models/vit_image_embedding.pt";

	SamPredictor predictor(1024, defaultMobileSamPredictor,
			       defaultVitImageEmbedding);
	auto maskInput =
		torch::zeros({ 1, 1, 256, 256 }, torch::dtype(torch::kFloat32));

	// point coords for picture2.jpg
	// std::vector<float> pointCoords = { 371.0f, 190.0f, 374.0f, 190.0f, 365.0f, 190.0f, 371.0f, 190.0f, 371.0f, 190.0f };

	cv::Mat jpg = cv::imread(defaultImagePath, cv::IMREAD_COLOR);

	predictor.setImage(jpg);

	// `pointLabels`: Labels for the sparse input prompts.
	// 0 is a negative input point,
	// 1 is a positive input point,
	// 2 is a top-left box corner,
	// 3 is a bottom-right box corner,
	// and -1 is a padding point. If there is no box input,
	// a single padding point with label -1 and coordinates (0.0, 0.0) should be concatenated.
	std::vector<float> pointLabels = {
		2.0f, 3.0f,
	};

	while (pointLabels.size() < 5) {
		pointLabels.emplace_back(-1.0f);
	}
	bool hasMaskInput = false;

	// Define 5 points manually
	std::vector<std::pair<float, float> > points = {
		{ 59.0f, 260.0f },
		{ 191.0f, 443.0f },
	};
	while (points.size() < 5) {
		points.emplace_back(0.0f, 0.0f);
	}
	std::vector<std::pair<float, float> > transformedCoords =
		predictor.transform.applyCoords(
			points, { predictor.originalImageHeight,
				  predictor.originalImageWidth });

	// Convert the transformed coordinates back to a flat vector
	std::vector<float> flatTransformedCoords;
	for (const auto &coord : transformedCoords) {
		flatTransformedCoords.push_back(coord.first);
		flatTransformedCoords.push_back(coord.second);
	}
	assert(flatTransformedCoords.size() == 10);

	torch::Tensor pointCoordsTensor = torch::tensor(
		flatTransformedCoords, torch::dtype(torch::kFloat32));

	torch::Tensor pointLabelsTensor =
		torch::tensor(pointLabels, torch::dtype(torch::kFloat32));

	pointCoordsTensor =
		pointCoordsTensor.reshape({ 1, 5, 2 }).to(torch::kFloat32);
	pointLabelsTensor =
		pointLabelsTensor.reshape({ 1, 5 }).to(torch::kFloat32);

	/***
	 * run inference
	 */

	torch::Tensor masks, IOUPredictions, lowResMasks;
	std::tie(masks, IOUPredictions, lowResMasks) = predictor.predict(
		pointCoordsTensor, pointLabelsTensor, maskInput, hasMaskInput);

	visualizeResults(jpg, masks, IOUPredictions, pointCoordsTensor);
}
