#ifndef SAM_AUTOMATIC_MASK_GENERATOR_H
#define SAM_AUTOMATIC_MASK_GENERATOR_H

#include <torch/script.h>
#include <map>
#include <opencv2/opencv.hpp>
#include <string>
#include <vector>

#include "predictor.h"

class AutomaticMaskGenerator {
       public:
        AutomaticMaskGenerator(SamPredictor& predictor, int pointsPerSide,
                                  int pointsPerBatch, float predIouThresh,
                                  float stabilityScoreThresh,
                                  float stabilityScoreOffset,
                                  float boxNmsThresh, int cropNLayers,
                                  float cropNmsThresh, float cropOverlapRatio,
                                  int cropNPointsDownscaleFactor,
                                  const std::string& outputMode);

        std::vector<std::map<std::string, torch::Tensor>> generate(
            const cv::Mat& image);

       private:
        SamPredictor& predictor;
        int pointsPerBatch;
        float predIouThresh;
        float stabilityScoreThresh;
        float stabilityScoreOffset;
        float boxNmsThresh;
        int cropNLayers;
        float cropNmsThresh;
        float cropOverlapRatio;
        int cropNPointsDownscaleFactor;
        std::string outputMode;
        std::vector<torch::Tensor> pointGrids;

        void buildAllLayerPointGrids(int pointsPerSide);
        std::vector<std::map<std::string, torch::Tensor>> generateMasks(
            const cv::Mat& image);
        void processCrop(const cv::Mat& image, const std::vector<int>& cropBox,
                         int layerIdx, const std::vector<int>& origSize);
        std::vector<torch::Tensor> processBatch(
            const torch::Tensor& points, const std::vector<int>& imSize,
            const std::vector<int>& cropBox, const std::vector<int>& origSize);
        void postprocessSmallRegions(
            std::vector<std::map<std::string, torch::Tensor>>& maskData,
            int minArea, float nmsThresh);
};

#endif  // SAM_AUTOMATIC_MASK_GENERATOR_H
