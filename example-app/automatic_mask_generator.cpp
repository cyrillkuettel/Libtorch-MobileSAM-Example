//
// Created by cyrill on 18.06.2024.
//

#include <torch/script.h>
#include <opencv2/opencv.hpp>
#include <algorithm>
#include <cmath>
#include <map>
#include <string>
#include <vector>

#include "automatic_mask_generator.h"
#include "predictor.h"

AutomaticMaskGenerator::AutomaticMaskGenerator(SamPredictor& predictor, int pointsPerSide, int pointsPerBatch,
                                                     float predIouThresh, float stabilityScoreThresh, float stabilityScoreOffset,
                                                     float boxNmsThresh, int cropNLayers, float cropNmsThresh,
                                                     float cropOverlapRatio, int cropNPointsDownscaleFactor,
                                                     const std::string& outputMode)
    : predictor(predictor), pointsPerBatch(pointsPerBatch), predIouThresh(predIouThresh),
      stabilityScoreThresh(stabilityScoreThresh), stabilityScoreOffset(stabilityScoreOffset),
      boxNmsThresh(boxNmsThresh), cropNLayers(cropNLayers), cropNmsThresh(cropNmsThresh),
      cropOverlapRatio(cropOverlapRatio), cropNPointsDownscaleFactor(cropNPointsDownscaleFactor),
      outputMode(outputMode) {
    buildAllLayerPointGrids(pointsPerSide);
}

std::vector<std::map<std::string, torch::Tensor>> AutomaticMaskGenerator::generate(const cv::Mat& image) {
    return generateMasks(image);
}

void AutomaticMaskGenerator::buildAllLayerPointGrids(int pointsPerSide) {
    // Implement the logic to build point grids for each layer based on the pointsPerSide and other parameters
    pointGrids = build_all_layer_point_grids(pointsPerSide, cropNLayers, cropNPointsDownscaleFactor);
}

std::vector<std::map<std::string, torch::Tensor>> AutomaticMaskGenerator::generateMasks(const cv::Mat& image) {
    std::vector<int> origSize = {image.rows, image.cols};
    auto [cropBoxes, layerIdxs] = generateCropBoxes(origSize, cropNLayers, cropOverlapRatio);

    std::vector<std::map<std::string, torch::Tensor>> maskData;
    for (size_t i = 0; i < cropBoxes.size(); ++i) {
        processCrop(image, cropBoxes[i], layerIdxs[i], origSize);
    }

    // Remove duplicate masks between crops using NMS
    if (cropBoxes.size() > 1) {
        applyNMS(maskData, cropNmsThresh);
    }

    // Convert data to numpy format if needed
    convertDataToNumpy(maskData);

    return maskData;
}

void AutomaticMaskGenerator::processCrop(const cv::Mat& image, const std::vector<int>& cropBox,
                                            int layerIdx, const std::vector<int>& origSize) {
    cv::Rect rect(cropBox[0], cropBox[1], cropBox[2] - cropBox[0], cropBox[3] - cropBox[1]);
    cv::Mat croppedImage = image(rect);
    std::vector<int> croppedImSize = {croppedImage.rows, croppedImage.cols};
    predictor.setImage(croppedImage);

    torch::Tensor pointsScale = torch::from_blob(croppedImSize.data(), {1, 2}).flip(1);
    torch::Tensor pointsForImage = pointGrids[layerIdx] * pointsScale;

    for (const auto& points : batchIterator(pointsPerBatch, pointsForImage)) {
        auto batchData = processBatch(points, croppedImSize, cropBox, origSize);
        maskData.insert(maskData.end(), batchData.begin(), batchData.end());
    }

    predictor.resetImage();

    // Remove duplicates within this crop using NMS
    applyNMS(maskData, boxNmsThresh);

    // Uncrop the boxes, points, and crop boxes back to the original image frame
    uncrop(maskData, cropBox, origSize);
}

std::vector<torch::Tensor> AutomaticMaskGenerator::processBatch(const torch::Tensor& points, const std::vector<int>& imSize,
                                                                   const std::vector<int>& cropBox, const std::vector<int>& origSize) {
    std::vector<torch::Tensor> batchData;

    auto transformedPoints = predictor.transformPoints(points, imSize);
    auto inPoints = transformedPoints.to(predictor.device);
    auto inLabels = torch::ones({inPoints.size(0)}, torch::TensorOptions().dtype(torch::kInt).device(inPoints.device()));

    auto [masks, iouPreds] = predictor.predictTorch(inPoints.unsqueeze(1), inLabels.unsqueeze(1), true, true);

    for (size_t i = 0; i < masks.size(0); ++i) {
        std::map<std::string, torch::Tensor> data;
        data["masks"] = masks[i];
        data["iou_preds"] = iouPreds[i];
        data["points"] = points[i];

        if (data["iou_preds"].item<float>() > predIouThresh) {
            batchData.push_back(data);
        }
    }

    return batchData;
}

void AutomaticMaskGenerator::postprocessSmallRegions(std::vector<std::map<std::string, torch::Tensor>>& maskData, int minArea, float nmsThresh) {
    for (auto& data : maskData) {
        auto mask = data["masks"];
        auto rle = maskToRLE(mask);

        auto filteredRLE = removeSmallRegions(rle, minArea);
        data["segmentation"] = rleToMask(filteredRLE);

        applyNMS(maskData, nmsThresh);
    }
}

void AutomaticMaskGenerator::applyNMS(std::vector<std::map<std::string, torch::Tensor>>& maskData, float nmsThresh) {
    std::vector<torch::Tensor> boxes, scores;
    for (const auto& data : maskData) {
        boxes.push_back(data.at("boxes"));
        scores.push_back(data.at("iou_preds"));
    }

    auto keep = batched_nms(torch::stack(boxes), torch::stack(scores), torch::zeros_like(scores[0]), nmsThresh);

    std::vector<std::map<std::string, torch::Tensor>> filteredData;
    for (const auto& idx : keep) {
        filteredData.push_back(maskData[idx.item<int>()]);
    }
    maskData = std::move(filteredData);
}


void AutomaticMaskGenerator::convertDataToNumpy(std::vector<std::map<std::string, torch::Tensor>>& maskData) {
    // Assuming conversion of torch::Tensor to numpy array or other required format
    for (auto& data : maskData) {
        for (auto& item : data) {
            item.second = item.second.cpu(); // Ensure tensors are on CPU
        }
    }
}

void AutomaticMaskGenerator::uncrop(std::vector<std::map<std::string, torch::Tensor>>& maskData, const std::vector<int>& cropBox, const std::vector<int>& origSize) {
    for (auto& data : maskData) {
        data["boxes"] = uncrop_boxes_xyxy(data["boxes"], cropBox);
        data["points"] = uncrop_points(data["points"], cropBox);
        data["crop_boxes"] = torch::tensor({cropBox}, torch::TensorOptions().dtype(torch::kInt));
    }
}
