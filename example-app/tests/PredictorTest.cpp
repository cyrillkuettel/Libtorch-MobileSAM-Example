#include "UnitTest.h"

TEST_CASE("", "Test predict with input point") {
        SECTION("Test Elephants detected") {
                SamPredictor predictor = setupPredictor();

                const AppConfig config = {
                    {{130.0f, 80.0f}},
                    {1.0f},  // top left, bottom right
                    "images/elephants.jpg",
                    false,
                };
                fs::path sourceDir = fs::path(__FILE__).parent_path();
                fs::path defaultImagePath = sourceDir / config.defaultImagePath;

                REQUIRE(fs::exists(defaultImagePath));
                validateAppConfig(config);

                torch::Tensor maskInput = torch::zeros(
                    {1, 1, 256, 256}, torch::dtype(torch::kFloat32));
                cv::Mat jpg =
                    imread(defaultImagePath.string(), cv::IMREAD_COLOR);
                predictor.setImage(jpg);

                std::vector<std::pair<float, float>> points;

                auto [pointCoordsTensor, pointLabelsTensor] =
                    computePointsAndLabels(config, jpg, predictor, "");
                pointCoordsTensor =
                    pointCoordsTensor.reshape({1, 5, 2}).to(torch::kFloat32);
                pointLabelsTensor =
                    pointLabelsTensor.reshape({1, 5}).to(torch::kFloat32);
                bool hasMaskInput = false;

                auto start = std::chrono::high_resolution_clock::now();
                torch::Tensor masks;
                torch::Tensor IOUPredictions;
                torch::Tensor lowResMasks;
                std::tie(masks, IOUPredictions, lowResMasks) =
                    predictor.predict(pointCoordsTensor, pointLabelsTensor,
                                      maskInput, hasMaskInput);
                auto stop = std::chrono::high_resolution_clock::now();
                auto duration =
                    std::chrono::duration_cast<std::chrono::milliseconds>(
                        stop - start);
                std::cout << "predictor.predict duration: " << duration.count()
                          << "ms\n";

                std::vector<cv::Mat> outImages = createInMemoryImages(
                    jpg, masks, IOUPredictions, pointCoordsTensor);

                REQUIRE(outImages.size() == 1);
                const fs::path expectedElephantOutput =
                    sourceDir / "images/expected_elephant_output.png";

                cv::Mat expectedImage =
                    imread(expectedElephantOutput.string(), cv::IMREAD_COLOR);
                REQUIRE(!expectedImage.empty());


                REQUIRE(outImages[0].rows == expectedImage.rows);
                REQUIRE(outImages[0].cols == expectedImage.cols);
                REQUIRE(outImages[0].type() == expectedImage.type());
        }
}
