#include "predictor.h"


void SamPredictor::set_image(const cv::Mat &jpg) {

    cv::Mat img;
    cv::cvtColor(jpg.clone(), img, cv::COLOR_BGR2RGB);

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
    set_torch_image(inputTensor);
}

void SamPredictor::reset_image() {
    // Implementation to reset the state as needed
}

void SamPredictor::set_torch_image(torch::Tensor &inputTensor) {
/*
    assert (
            len(transformed_image.shape) == 4
            and transformed_image.shape[1] == 3
            and max(*transformed_image.shape[2:]) == self.model.image_encoder.img_size
    ), f"set_torch_image input must be BCHW with long side {self.model.image_encoder.img_size}."
    self.reset_image()

    self.original_size = original_image_size
    self.input_size = tuple(transformed_image.shape[-2:])
#import pdb; pdb.set_trace()
    input_image = self.model.preprocess(transformed_image)
    self.features = self.model.image_encoder(input_image)
    self.is_image_set = True
*/

    if (!(inputTensor.sizes().size() == 4 && inputTensor.size(1) == 3)) {
        throw std::runtime_error(
                "set_torch_image input must be BCHW with long side");
    }

    torch::Tensor tensor_pixel_mean = torch::tensor(pixel_mean);
    torch::Tensor tensor_pixel_std = torch::tensor(pixel_std);

    // Reshape mean and std tensors for broadcasting
    tensor_pixel_mean = tensor_pixel_mean.view({1, 3, 1, 1});
    tensor_pixel_std = tensor_pixel_std.view({1, 3, 1, 1});
    inputTensor = (inputTensor - tensor_pixel_mean) / tensor_pixel_std;

    // Pad
    // fixme: is this really the img.cols???
    // this is the already transformed image,
    // therefore this would be 1024 - 1024 = 0
    int padh = 1024 - input_size.first;
    int padw = 1024 - input_size.second;
    torch::nn::functional::PadFuncOptions padOptions({0, padw, 0, padh});
    inputTensor = torch::nn::functional::pad(inputTensor, padOptions);

    std::vector<torch::jit::IValue> inputs{inputTensor};

    /** image_encoder (ImageEncoderViT, imageEmbeddingModel):
     * The backbone used to encode the image into image embeddings
     * that allow for efficient mask prediction */
    features = imageEmbeddingModel.forward(inputs).toTensor();
    is_image_set = true;
}


std::tuple<torch::Tensor, torch::Tensor, torch::Tensor>
SamPredictor::predict(const torch::Tensor &pointCoords, const torch::Tensor &pointLabels) {


    // int mask_input_size = [4 * x for x in embed_size] == [256, 256]

    auto maskInput =
            torch::zeros({1, 1, 256, 256}, torch::dtype(torch::kFloat32));

    //btw. why is the tensor size [1500, 2250]?
    auto origImgSize =
            torch::tensor({originalHeight, originalWidth},
                          torch::dtype(torch::kFloat32));

    auto hasMaskInput = torch::tensor({0}, torch::dtype(torch::kFloat32));
    std::vector<torch::jit::IValue> inputs2;
    inputs2.emplace_back(features); // image_embeddings
    inputs2.emplace_back(pointCoords);
    inputs2.emplace_back(pointLabels);
    inputs2.emplace_back(maskInput);
    inputs2.emplace_back(hasMaskInput);
    inputs2.emplace_back(origImgSize);

    auto modeloutput = predictorModel.forward(inputs2);
    // Accessing output tensors

    auto outputs = modeloutput.toTuple()->elements();

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

