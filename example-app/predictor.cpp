#include "predictor.h"

void SamPredictor::setImage(const cv::Mat& image) {
        if (image.empty()) {
                std::cout << "imread(): failed: Image not found" << std::endl;
        }

        cv::Mat img;
        cv::cvtColor(image.clone(), img, cv::COLOR_BGR2RGB);

        originalImageHeight = img.rows;
        originalImageWidth = img.cols;

        // img_converted.convertTo(img, CV_8UC3) seems redundant in this context
        // because the image, once loaded and potentially color-converted,
        // is already in the CV_8UC3 format. But it pays to be paranoid...
        img.convertTo(img, CV_8UC3);
        transform.applyImage(img);
        std::map<int, std::string> depthMap = {
            {CV_8U, "CV_8U"},   {CV_8S, "CV_8S"},   {CV_16U, "CV_16U"},
            {CV_16S, "CV_16S"}, {CV_32S, "CV_32S"}, {CV_32F, "CV_32F"},
            {CV_64F, "CV_64F"}};

        int depth = img.depth();
        std::string depthStr =
            depthMap.count(depth) ? depthMap[depth] : "Unknown";

        auto inputTensor =
            torch::from_blob(img.data, {img.rows, img.cols, 3}, torch::kByte);

        inputTensor = inputTensor.permute({2, 0, 1}).contiguous();
        inputTensor = inputTensor.unsqueeze(0);
        setTorchImage(inputTensor);
}

void SamPredictor::setTorchImage(torch::Tensor& inputTensor) {
        if (!(inputTensor.sizes().size() == 4 && inputTensor.size(1) == 3)) {
                throw std::runtime_error(
                    "setTorchImage input must be BCHW with long side");
        }
        preProcess(inputTensor);

        // Print the width and height of the tensor
        int64_t height = inputTensor.size(2);
        int64_t width = inputTensor.size(3);

        std::vector<torch::jit::IValue> inputs{inputTensor};

        /** image_encoder (ImageEncoderViT, imageEmbeddingModel):
     * The backbone used to encode the image into image embeddings
     * that allow for efficient mask prediction */
        features = imageEmbeddingModel.forward(inputs).toTensor();
        isImageSet = true;
}

torch::Tensor customPad(const torch::Tensor& self, at::IntArrayRef pad,
                        const at::Scalar& value) {
        TORCH_CHECK(pad.size() % 2 == 0,
                    "Length of pad must be even but instead it equals ",
                    pad.size());

        auto input_sizes = self.sizes();
        auto l_inp = self.dim();

        auto l_pad = pad.size() / 2;
        auto l_diff = l_inp - l_pad;
        TORCH_CHECK(l_inp >= (int64_t)l_pad,
                    "Length of pad should be no more than twice the number of "
                    "dimensions of the input. Pad length is ",
                    pad.size(), "while the input has ", l_inp, "dimensions.");

        std::vector<int64_t> new_shape;

        bool all_pads_non_positive = true;

        auto c_input = self;
        for (const auto i : c10::irange(l_diff, l_inp)) {
                auto pad_idx = 2 * (l_inp - i - 1);
                if (pad[pad_idx] < 0) {
                        c_input = c_input.narrow(
                            i, -pad[pad_idx], c_input.size(i) + pad[pad_idx]);
                } else if (pad[pad_idx] != 0) {
                        all_pads_non_positive = false;
                }
                if (pad[pad_idx + 1] < 0) {
                        c_input = c_input.narrow(
                            i, 0, c_input.size(i) + pad[pad_idx + 1]);
                } else if (pad[pad_idx + 1] != 0) {
                        all_pads_non_positive = false;
                }
        }

        // if none of the pads are positive we can optimize and just return the result
        // of calling .narrow() on the input
        if (all_pads_non_positive) {
                return c_input.clone();
        }

        for (size_t i = 0; i < (size_t)l_diff; i++) {
                new_shape.emplace_back(input_sizes[i]);
        }

        for (const auto i : c10::irange((size_t)l_pad)) {
                auto pad_idx = pad.size() - ((i + 1) * 2);
                auto new_dim =
                    input_sizes[l_diff + i] + pad[pad_idx] + pad[pad_idx + 1];
                TORCH_CHECK(new_dim > 0, "The input size ",
                            input_sizes[l_diff + i], ", plus negative padding ",
                            pad[pad_idx], " and ", pad[pad_idx + 1],
                            " resulted in a negative output size, "
                            "which is invalid. Check dimension ",
                            l_diff + i, " of your input.");
                new_shape.emplace_back(new_dim);
        }

        at::Tensor output;
        const auto memory_format = self.suggest_memory_format();
        if (self.is_quantized()) {
                const auto qscheme = self.qscheme();
                TORCH_CHECK(qscheme == at::kPerTensorAffine ||
                                qscheme == at::kPerTensorSymmetric,
                            "Only per-tensor padding is supported.");
                output = at::_empty_affine_quantized(
                    new_shape, self.options().memory_format(memory_format),
                    self.q_scale(), self.q_zero_point(), c10::nullopt);
        } else {
                output = at::empty(new_shape,
                                   self.options().memory_format(memory_format));
        }
        output.fill_(value);

        auto c_output = output;
        for (const auto i : c10::irange(l_diff, l_inp)) {
                auto pad_idx = 2 * (l_inp - i - 1);
                if (pad[pad_idx] > 0) {
                        c_output = c_output.narrow(
                            i, pad[pad_idx], c_output.size(i) - pad[pad_idx]);
                }
                if (pad[pad_idx + 1] > 0) {
                        c_output = c_output.narrow(
                            i, 0, c_output.size(i) - pad[pad_idx + 1]);
                }
        }
        c_output.copy_(c_input);
        return output;
}

void SamPredictor::preProcess(torch::Tensor& inputTensor) {
        torch::Tensor tensor_pixel_mean = torch::tensor(pixelMean);
        torch::Tensor tensor_pixel_std = torch::tensor(pixelStd);

        // Normalize pixel values and pad to a square input.
        // Normalize colors, Reshape mean and std tensors for broadcasting
        tensor_pixel_mean = tensor_pixel_mean.view({1, 3, 1, 1});
        tensor_pixel_std = tensor_pixel_std.view({1, 3, 1, 1});
        inputTensor = (inputTensor - tensor_pixel_mean) / tensor_pixel_std;

        // Optionally, print the datatypes of each tensor
        std::cout << "inputTensor dtype: " << inputTensor.scalar_type()
                  << std::endl;
        std::cout << "tensor_pixel_mean dtype: "
                  << tensor_pixel_mean.scalar_type() << std::endl;
        std::cout << "tensor_pixel_std dtype: "
                  << tensor_pixel_std.scalar_type() << std::endl;

        int64_t h = inputTensor.size(2);
        int64_t w = inputTensor.size(3);
        int64_t padh = inputSize.first - h;
        int64_t padw = inputSize.second - w;
        std::cout << "padh: " << padh << std::endl;
        std::cout << "padw: " << padw << std::endl;

        // padding_left,padding_right, padding_top,padding_bottom)

        torch::nn::functional::PadFuncOptions padOptions({0, padw, 0, padh});
        // inputTensor = torch::nn::functional::pad(inputTensor, padOptions);

        std::vector<int64_t> padding = {0, padw, 0, padh};
        c10::IntArrayRef pad(padding);

        inputTensor = customPad(inputTensor, pad, 0);

        //	 torch::Tensor equal = torch::eq(otherTensor, inputTensor);
        //	 std::cout << "Element-wise equality:\n" << equal << std::endl;
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> SamPredictor::predict(
    const torch::Tensor& pointCoordsTensor,
    const torch::Tensor& pointLabelsTensor, const torch::Tensor& maskInput,
    bool maskInputBool) {
        int val = maskInputBool ? 1 : 0;

        // XXX this syntax does not work on Android:
        // it fails with dreaded linker error: "undefined reference to `c10::operator<<(std::__ndk1::basic_ostream ..."
        // torch::Tensor hasMaskInput = torch::tensor({ val }, torch::dtype(torch::kFloat32));

        std::vector<int> hasMaskInputVec = {val};
        torch::Tensor hasMaskInput =
            torch::tensor(hasMaskInputVec, torch::dtype(torch::kFloat32));
        // torch::Tensor hasMaskInput = zTensor.toType(torch::kFloat);

        //btw. why is the tensor size [1500, 2250]?
        // not sure if I need static_cast here?
        std::vector<int> imgSize = {originalImageHeight, originalImageWidth};
        torch::Tensor origImgSize =
            torch::tensor(imgSize, torch::dtype(torch::kFloat32));

        std::vector<torch::jit::IValue> inputs2 = {
            features,  pointCoordsTensor, pointLabelsTensor,
            maskInput, hasMaskInput,      origImgSize};

        auto modelOutput = predictorModel.forward(inputs2);
        auto outputs = modelOutput.toTuple()->elements();

        // debugPrint(outputs);
        torch::Tensor masks = outputs[0].toTensor();
        torch::Tensor iou_predictions = outputs[1].toTensor();
        torch::Tensor low_res_masks = outputs[2].toTensor();

        return std::make_tuple(masks, iou_predictions, low_res_masks);
}

void SamPredictor::debugPrint(const c10::ivalue::TupleElements& outputs) const {
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
}

void SamPredictor::resetImage() {
        isImageSet = false;
        features = torch::Tensor();
        originalImageHeight = 0;
        originalImageWidth = 0;
}
