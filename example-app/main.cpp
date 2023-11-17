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

	// `pointLabels`: Labels for the sparse input prompts.
	// 0 is a negative input point,
	// 1 is a positive input point,
	// 2 is a top-left box corner,
	// 3 is a bottom-right box corner,
	// and -1 is a padding point. If there is no box input,
	// ka single padding point with label -1 and coordinates (0.0, 0.0) should be concatenated.
	std::vector<float> pointLabels = { 1.0f, -1.0f, -1.0f, -1.0f, -1.0f };

	// rest is zero
	std::vector<float> pointCoords = {
		20.0f, 20.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f
	};

	cv::Mat jpg = cv::imread(defaultImagePath, cv::IMREAD_COLOR);

	predictor.setImage(jpg);

	torch::Tensor masks, IOUPredictions, lowResMasks;
	std::tie(masks, IOUPredictions, lowResMasks) =
		predictor.predict(pointCoords, pointLabels, maskInput, false);

	torch::Tensor pointCoordsTensor = torch::tensor(
		pointCoords, torch::dtype(torch::kFloat32));

	torch::Tensor pointLabelsTensor =
		torch::tensor(pointLabels, torch::dtype(torch::kFloat32));

	pointCoordsTensor =
		pointCoordsTensor.reshape({ 1, 5, 2 }).to(torch::kFloat32);

	visualizeResults(jpg, masks, IOUPredictions, pointCoordsTensor );
}
