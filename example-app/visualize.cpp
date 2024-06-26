#include <cstdlib>
#include <iostream>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>

#include "visualize.hpp"

void printMatType(const cv::Mat& mat) {
        int type = mat.type();
        std::string typeStr;

        switch (type) {
                case CV_8U:
                        typeStr = "8U";
                        break;
                case CV_8S:
                        typeStr = "8S";
                        break;
                case CV_16U:
                        typeStr = "16U";
                        break;
                case CV_16S:
                        typeStr = "16S";
                        break;
                case CV_32S:
                        typeStr = "32S";
                        break;
                case CV_32F:
                        typeStr = "32F";
                        break;
                case CV_64F:
                        typeStr = "64F";
                        break;
                default:
                        typeStr = "Unknown";
                        break;
        }

        std::cout << "Matrix type: " << typeStr << std::endl;
#ifdef __ANDROID
        __android_log_print(ANDROID_LOG_INFO, "predictor.cpp",
                            "Matrix type: %s", typeStr.c_str());
#endif
}

cv::Mat tensorToMat(torch::Tensor tensor) {
        tensor = tensor.squeeze().detach();
        if (tensor.dtype() != torch::kU8) {
                tensor = tensor.to(torch::kU8);
        }

        cv::Mat mat(tensor.size(0), tensor.size(1), CV_8UC1,
                    tensor.data_ptr<uchar>());
        return mat.clone();  // Clone to safely detach from the tensor data
}

void showMask(const torch::Tensor& mask, cv::Mat& image) {
        auto start = std::chrono::high_resolution_clock::now();
        cv::Scalar color = cv::Scalar(255, 127, 0);  // BGR color

        cv::Mat maskMat = tensorToMat(mask);

        double alpha = 0.8;  // Transparency factor

        // this inverts the mask
        // These values come just from trial and error
        // This can probably be done better
        cv::threshold(maskMat, maskMat, 127, 255, cv::THRESH_BINARY_INV);

        printMatType(maskMat);

        // Apply color only to the masked areas
        for (int y = 0; y < image.rows; y++) {
                for (int x = 0; x < image.cols; x++) {
                        if (maskMat.at<uchar>(y, x)) {
                                for (int c = 0; c < 3; c++) {
                                        image.at<cv::Vec3b>(y, x)[c] =
                                            static_cast<uchar>(
                                                alpha * color[c] +
                                                (1 - alpha) *
                                                    image.at<cv::Vec3b>(y,
                                                                        x)[c]);
                                }
                        }
                }
        }

        auto stop = std::chrono::high_resolution_clock::now();
        auto duration =
            std::chrono::duration_cast<std::chrono::milliseconds>(stop - start);
        std::cout << "showMask duration: " << duration.count() << "ms\n";
}

// todo: write function to show box and points that is not buggy
void showPoints(const torch::Tensor& coords, cv::Mat& image, int markerSize) {
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
                           cv::Scalar(0, 255, 0),
                           -1);  // Green color for points
        }
}

// Main visualization function
void visualizeResults(cv::Mat& image, const torch::Tensor& masks,
                      const torch::Tensor& scores,
                      const torch::Tensor& pointCoords) {


        std::vector<cv::Mat> outImages = createInMemoryImages(
            image, masks, scores, pointCoords);
        return saveAndDisplayImages(outImages, scores);

}

std::vector<cv::Mat> createInMemoryImages(cv::Mat& image,
                                          const torch::Tensor& masks,
                                          const torch::Tensor& scores,
                                          const torch::Tensor& pointCoords) {
        std::vector<cv::Mat> outImages;
        showPoints(pointCoords, image, 5);
        for (int i = 0; i < masks.size(0); i++) {
                cv::Mat displayImage = image.clone();
                showMask(masks[i], displayImage);
                outImages.push_back(displayImage);
        }
        return outImages;
}

void saveAndDisplayImages(const std::vector<cv::Mat>& inMemoryImages,
                          const torch::Tensor& scores) {
        for (size_t i = 0; i < inMemoryImages.size(); i++) {
                // Generate filename with timestamp
                auto now = std::chrono::system_clock::now();
                auto now_c = std::chrono::system_clock::to_time_t(now);
                auto milliseconds =
                    std::chrono::duration_cast<std::chrono::milliseconds>(
                        now.time_since_epoch()) %
                    1000;

                std::tm tm = *std::localtime(&now_c);
                std::ostringstream oss;
                oss << std::put_time(&tm, "%Y%m%d_%H%M%S") << '_'
                    << std::setfill('0') << std::setw(3)
                    << milliseconds.count();
                std::string timestamp = oss.str();
                std::string filename = "MobileSAM" + std::to_string(i + 1) +
                                       "_" + timestamp + ".png";

                // Save output to png
                std::vector<int> compressionParams = {
                    cv::IMWRITE_PNG_COMPRESSION, 0};
                cv::imwrite(filename, inMemoryImages[i], compressionParams);

                // Print saved image path
                std::filesystem::path actualpath =
                    std::filesystem::absolute(filename);
                std::cout << "Saved output image: " << actualpath << std::endl;

                // Display image
                cv::imshow(
                    "Visualization - Mask " + std::to_string(i + 1) +
                        ", Score: " + std::to_string(scores[i].item<float>()),
                    inMemoryImages[i]);
                cv::waitKey(0);  // Wait for a key press
        }
}
