#include "predictor.h"


void SamPredictor::setImage(const cv::Mat &image) {

    cv::Mat img;
    cv::cvtColor(image.clone(), img, cv::COLOR_BGR2RGB);

    originalHeight = img.cols;
    originalWidth = img.rows;

    // img_converted.convertTo(img, CV_8UC3) seems redundant in this context
    // because the image, once loaded and potentially color-converted,
    // is already in the CV_8UC3 format. But it pays to be paranoid...
    img.convertTo(img, CV_8UC3);
    resizeLongestSide.apply_image(img);

    auto inputTensor = torch::from_blob(img.data, {img.rows, img.cols, 3},
                                        torch::kByte);

    inputTensor = inputTensor.permute({2, 0, 1}).contiguous();
    inputTensor = inputTensor.unsqueeze(0);
    // set originalWidth and originalHeight as param
    setTorchImage(inputTensor);
}

void SamPredictor::setTorchImage(torch::Tensor &inputTensor) {

    if (!(inputTensor.sizes().size() == 4 && inputTensor.size(1) == 3)) {
        throw std::runtime_error(
                "setTorchImage input must be BCHW with long side");
    }
    preProcess(inputTensor);

    std::vector<torch::jit::IValue> inputs{inputTensor};

    /** image_encoder (ImageEncoderViT, imageEmbeddingModel):
     * The backbone used to encode the image into image embeddings
     * that allow for efficient mask prediction */
    features = imageEmbeddingModel.forward(inputs).toTensor();
    isImageSet = true;
}

void SamPredictor::preProcess(torch::Tensor &inputTensor) {

    torch::Tensor tensor_pixel_mean = torch::tensor(pixel_mean);
    torch::Tensor tensor_pixel_std = torch::tensor(pixel_std);
    // Normalize pixel values and pad to a square input.
    // Normalize colors, Reshape mean and std tensors for broadcasting
    tensor_pixel_mean = tensor_pixel_mean.view({1, 3, 1, 1});
    tensor_pixel_std = tensor_pixel_std.view({1, 3, 1, 1});
    inputTensor = (inputTensor - tensor_pixel_mean) / tensor_pixel_std;

    // Pad
    // fixme: is this really the img.cols???
    // this is the already transformed image,
    // therefore this would be 1024 - 1024 = 0

    int64_t h = inputTensor.size(2);
    int64_t w = inputTensor.size(3);

    int64_t padh = inputSize.first - h;
    int64_t padw = inputSize.second - w;

    std::cout << "padh: " << padh << std::endl;
    std::cout << "padw: " << padw << std::endl;

    torch::nn::functional::PadFuncOptions padOptions({0, padw, 0, padh});
    torch::nn::functional::pad(inputTensor, padOptions);
}



std::tuple<torch::Tensor, torch::Tensor, torch::Tensor>
SamPredictor::predict(const std::vector<std::vector<float>> &pointCoordsVec, const std::vector<int> &pointLabelsVec,
                      const torch::Tensor &maskInput, bool maskInputBool) {

    int val = maskInputBool ? 1 : 0;
    torch::Tensor hasMaskInput = torch::tensor({val}, torch::dtype(torch::kFloat32));

    std::vector<float> flattenedPointCoordsVec = linearize(pointCoordsVec);
    torch::Tensor pointCoordsTensor =
            torch::tensor(flattenedPointCoordsVec, torch::dtype(torch::kFloat32));

    torch::Tensor pointLabelsTensor =
            torch::tensor(pointLabelsVec, torch::dtype(torch::kFloat32));

    pointCoordsTensor =
            pointCoordsTensor.reshape({1, 5, 2}).to(torch::kFloat32);
    pointLabelsTensor =
            pointLabelsTensor.reshape({1, 5}).to(torch::kFloat32);

    //btw. why is the tensor size [1500, 2250]?
    torch::Tensor origImgSize =
            torch::tensor({originalHeight, originalWidth},
                          torch::dtype(torch::kFloat32));

    std::vector<torch::jit::IValue> inputs2 = {features, pointCoordsTensor, pointLabelsTensor, maskInput, hasMaskInput,
                                               origImgSize};

    auto modelOutput = predictorModel.forward(inputs2);
    auto outputs = modelOutput.toTuple()->elements();

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

void SamPredictor::resetImage() {
    isImageSet = false;
    features = torch::Tensor();
    originalHeight = 0;
    originalWidth = 0;
}
