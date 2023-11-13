# Libtorch MobileSAM 

A minimal example of how to use Libtorch with MobileSAM and OpenCV in the same project.

[Screencast from 13.11.2023 03:25:12.webm](https://github.com/cyrillkuettel/Libtorch-MobileSAM-Example/assets/36520284/f5311c46-644f-45a7-adf3-a60bc853f4a9)


## Quick Start

###  dependencies
```console
 sudo apt install build-essential cmake git pkg-config libgtk-3-dev libavcodec-dev libavformat-dev libswscale-dev libv4l-dev  libxvidcore-dev libx264-dev libjpeg-dev libpng-dev libtiff-dev gfortran openexr libatlas-base-dev python3-dev python3-numpy  libtbb2 libtbb-dev libdc1394-22-dev
```
Note: You need to download Libtorch 1.13.0 and install OpenCV 4.5.4.

(Other versions _might_ work, but have not been tested untested)
### Libtorch 
The project expects `libtorch/` in the top-level directory. I have not included this because its 727MB. 

Just download [this version from pytorch.org](https://download.pytorch.org/libtorch/cpu/libtorch-cxx11-abi-shared-with-deps-1.13.0%2Bcpu.zip),  rename the folder to 'libtorch' and put it in the repository at top level.

```bash
wget https://download.pytorch.org/libtorch/cpu/libtorch-cxx11-abi-shared-with-deps-1.13.0%2Bcpu.zip
unzip libtorch-cxx11-abi-shared-with-deps-1.13.0+cpu.zip # important that it's the  `cxx11 ABI` version, works with OpenCV)
```

### OpenCV 
Install OpenCV for your operating system. 
#### Linux


```bash
mkdir ~/opencv && cd opencv
git clone https://github.com/opencv/opencv_contrib.git
git clone https://github.com/opencv/opencv.git
cd opencv
mkdir build 
cd build

cmake -D CMAKE_BUILD_TYPE=RELEASE \
	-D CMAKE_INSTALL_PREFIX=/usr/local \
	-D INSTALL_C_EXAMPLES=ON \
	-D INSTALL_PYTHON_EXAMPLES=ON \
	-D OPENCV_GENERATE_PKGCONFIG=ON \
	-D OPENCV_EXTRA_MODULES_PATH=../../opencv_contrib/modules \
        -D PYTHON_EXECUTABLE=/usr/bin/python3 \
	-D OPENCV_BUILD_3RDPARTY_LIBS=ON \
	-D BUILD_EXAMPLES=ON ..
  
make -j8
sudo make install
```


## Run

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

## Information on Libtorch
For first time install troubleshooting, see the [pytorch cppdocs](https://pytorch.org/cppdocs/installing.html), which this information is based on.
