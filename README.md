# Libtorch MobileSAM 

A minimal example of how ot use Libtorch with MobileSAM and OpenCV in the same project.

## Getting started
The project expects `libtorch/` in the top-level direcotry. I have not included this in the repository to keep it light. 

Download here (It's importatnt that you use the `cxx11 ABI` version, which works with OpenCV):

```bash
wget https://download.pytorch.org/libtorch/cpu/libtorch-cxx11-abi-shared-with-deps-1.13.0%2Bcpu.zip
unzip libtorch-cxx11-abi-shared-with-deps-1.13.0+cpu.zip
```
rename the folder to 'libtorch' if it isn't already. 


Then, in the example-app, the first time you might have to run these commands. 

 
```bash
mkdir build
cd build
cmake -DCMAKE_BUILD_TYPE=Debug -DCMAKE_PREFIX_PATH=/absolute/path/to/libtorch ..
cmake --build . --config Debug
```


For subsequent builds you can just type: 

```
make
```


## Information on Libtorch
For first time install troubleshooting, see the [pytorch cppdocs](https://pytorch.org/cppdocs/installing.html), which this information is based on.
