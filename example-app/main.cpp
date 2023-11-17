#include "predictor.h"
#include "visualize.cpp"

int main()
{
	std::string defaultImagePath =
		"/home/cyrill/pytorch/libtorch-opencv/example-app/images/picture2.jpg";
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
	// a single padding point with label -1 and coordinates (0.0, 0.0) should be concatenated.
	std::vector<float> pointLabels = { 1.0f, 1.0f, 1.0f, 1.0f, 1.0f };
	bool hasMaskInput = false;

	//std::vector<float> pointCoords = { 310.0f, 180.0f, 310.0f, 180.0f, 310.0f, 180.0f, 310.0f, 180.0f, 310.0f, 180.0f };

	// point coords for picture2.jpg
	std::vector<float> pointCoords = { 371.0f, 190.0f, 374.0f, 190.0f, 365.0f, 190.0f, 371.0f, 190.0f, 371.0f, 190.0f };

	cv::Mat jpg = cv::imread(defaultImagePath, cv::IMREAD_COLOR);

	predictor.setImage(jpg);

	torch::Tensor masks, IOUPredictions, lowResMasks;
	std::tie(masks, IOUPredictions, lowResMasks) = predictor.predict(
		pointCoords, pointLabels, maskInput, hasMaskInput);

	torch::Tensor pointCoordsTensor =
		torch::tensor(pointCoords, torch::dtype(torch::kFloat32));

	torch::Tensor pointLabelsTensor =
		torch::tensor(pointLabels, torch::dtype(torch::kFloat32));

	pointCoordsTensor =
		pointCoordsTensor.reshape({ 1, 5, 2 }).to(torch::kFloat32);

	visualizeResults(jpg, masks, IOUPredictions, pointCoordsTensor);
}
