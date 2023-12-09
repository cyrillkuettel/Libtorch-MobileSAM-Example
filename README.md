# Libtorch MobileSAM 


A C++ implementation of [MobileSAM](https://github.com/ChaoningZhang/MobileSAM).



https://github.com/cyrillkuettel/Libtorch-MobileSAM-Example/assets/36520284/73cbf5ce-e58a-45d5-ba36-9a78307ecb6a



## Description

The main goal of `Libtorch-MobileSAM-Example` is run traced or scripted Torchscript models. This provides the foundations for eventually running this on mobile devices (Pytorch Mobile).

## Todo

- [x] Feature: Refactor to be object oriented
- [x] Bug: Translate input coordinates to longest side for (1024x1024)
- [ ] Feature: Add ability to click with mouse 
- [ ] Bug: Fix the `drawPoints` function. 
- [ ] Feature: Implement [automatic_mask_generator](https://github.com/ChaoningZhang/MobileSAM/blob/master/mobile_sam/automatic_mask_generator.py)



## Quick Start
### Models

The models are included in the repo, alternatively, you can export them with this script [convert_pytorch_mobile.py](https://github.com/cmarschner/MobileSAM/blob/cmarschner/convert/scripts/convert_pytorch_mobile.py).

<details open>
    <summary>Models</summary>
    <ul>
        <li>
            <a href="https://github.com/cyrillkuettel/Libtorch-MobileSAM-Example/tree/master/example-app/models/">cpu_mobilesam_predictor_mobile_optimized_lite_interpreter.ptl</a>
        </li>
        <li>
            <a href="https://github.com/cyrillkuettel/Libtorch-MobileSAM-Example/blob/master/example-app/models/cpu_vit_image_embedding_mobile_optimized_lite_interpreter.ptl">pu_vit_image_embedding_mobile_optimized_lite_interpreter.ptl</a>
        </li>
        <li>
            <a href="https://github.com/cyrillkuettel/Libtorch-MobileSAM-Example/blob/master/example-app/models/mobilesam_predictor.pt">mobilesam_predictor.pt</a>
        </li>
        <li>
            <a href="https://github.com/cyrillkuettel/Libtorch-MobileSAM-Example/blob/master/example-app/models/vit_image_embedding.pt">vit_image_embedding.pt</a>
        </li>
    </ul>
</details>

<details>
<summary>Model input size. Source:
    <a href="https://github.com/ChaoningZhang/MobileSAM/blob/12d80d4e32b277de299130d8ce28cc949fb54b6c/notebooks/onnx_model_example.ipynb">notebooks/onnx_model_example.ipynb</a></summary>
    <ul>
        <li>
            `image_embeddings`: The image embedding from predictor.get_image_embedding(). Has a batch index of length 1.
        </li>
        <li>
            `point_coords`: Coordinates of sparse input prompts, corresponding to both point inputs and box inputs.
            Boxes
            are encoded using two points, one for the top-left corner and one for the bottom-right corner. Coordinates
            must already be transformed to long-side 1024. Has a batch index of length 1.
        </li>
        <li>
            `point_labels`: Labels for the sparse input prompts. 0 is a negative input point, 1 is a positive input
            point,
            2 is a top-left box corner, 3 is a bottom-right box corner, and -1 is a padding point. If there is no box
            input, a single padding point with label -1 and coordinates (0.0, 0.0) should be concatenated.
        </li>
        <li>
            `mask_input`: A mask input to the model with shape 1x1x256x256. This must be supplied even if there is no
            mask
            input. In this case, it can just be zeros.
        </li>
        <li>
            `has_mask_input`: An indicator for the mask input. 1 indicates a mask input, 0 indicates no mask input.
        </li>
        <li>
            `orig_im_size`: The size of the input image in (H,W) format, before any transformation.
        </li>
    </ul>
</details>



##  Dependencies
### Linux: 
```console
 sudo apt install build-essential cmake git pkg-config libgtk-3-dev libavcodec-dev libavformat-dev libswscale-dev libv4l-dev  libxvidcore-dev libx264-dev libjpeg-dev libpng-dev libtiff-dev gfortran openexr libatlas-base-dev python3-dev python3-numpy  libtbb2 libtbb-dev libdc1394-22-dev
```

### MacOS: 
```console
brew install cmake pkg-config jpeg libpng libtiff openexr eigen tbb
```
Note: Tested with Libtorch 1.13.0+ and OpenCV 4.5.4+

(Other versions _might_ work, but have not been tested untested)

### Libtorch dependency
The project expects `libtorch/` in the top-level directory. I have not included this because its 727MB. 

#### Mac M1 Chips

Pre-built binaries of pytorch for for Mac M2 can be found here [libtorch-mac-m1/releases](https://github.com/mlverse/libtorch-mac-m1/releases) (no official builds at the point of writing this.) 
Rename the folder to 'libtorch' and put it in the top-level directory of the repository.
#### Linux

Just download [this version from pytorch.org](https://download.pytorch.org/libtorch/cpu/libtorch-cxx11-abi-shared-with-deps-1.13.0%2Bcpu.zip), rename the folder to 'libtorch' and put it in the repository at top level.

```bash
wget https://download.pytorch.org/libtorch/cpu/libtorch-cxx11-abi-shared-with-deps-1.13.0%2Bcpu.zip
unzip libtorch-cxx11-abi-shared-with-deps-1.13.0+cpu.zip # important that it's the  `cxx11 ABI` version, works with OpenCV)
```

### OpenCV dependency
Install OpenCV for your operating system. 
#### Unix/MacOS
I did it like this: 

```bash
mkdir ~/opencv && cd ~/opencv
git clone https://github.com/opencv/opencv_contrib.git
git clone https://github.com/opencv/opencv.git
cd opencv
mkdir build 
cd build

cmake -D CMAKE_BUILD_TYPE=RELEASE \
	-D CMAKE_INSTALL_PREFIX=/usr/local \
	-D INSTALL_C_EXAMPLES=OFf \
	-D INSTALL_PYTHON_EXAMPLES=OFF \
	-D OPENCV_GENERATE_PKGCONFIG=ON \
	-D OPENCV_EXTRA_MODULES_PATH=../../opencv_contrib/modules \
        -D PYTHON_EXECUTABLE=/usr/bin/python3 \
	-D OPENCV_BUILD_3RDPARTY_LIBS=ON \
	-D BUILD_EXAMPLES=ON ..
  
make -j8
sudo make install
```


## Run
### Run from command line

Only first time: (Note the two dots at the end)
```bash
cd example-app
mkdir build
cd build
cmake -DCMAKE_BUILD_TYPE=Debug -DCMAKE_PREFIX_PATH=/absolute/path/to/libtorch ..
cmake --build . --config Debug
```

From now on, you can just type:

```
make
```

### Run from Clion 

1. File -> Open -> example-app
2. Open build settings (should open automatically): File -> Settings -> Build, Execution, Deployment -> CMake 
3. Delete contents of Cmake options, add this: `-DCMAKE_BUILD_TYPE=Debug -DCMAKE_PREFIX_PATH=/absolute/path/to/libtorch`

![Clion_setup.png](example-app%2FClion_setup.png)

## Information on Libtorch
For first time install troubleshooting, see the [pytorch cppdocs](https://pytorch.org/cppdocs/installing.html), which this information is based on.
