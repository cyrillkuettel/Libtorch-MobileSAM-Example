#include <torch/torch.h>
#include <iostream>
#include <opencv2/opencv.hpp>


int main() {
 std::cout << "The current OpenCV version is " << CV_VERSION << "\n";
  torch::Tensor tensor = torch::rand({2, 3});
  std::cout << tensor << std::endl;
}
