# Libtorch and Opencv

A minimal example of how ot use OpenCV and Libtorch in the same project.

The project expects libtorch/ in the top-level direcotry. I have not included this in the repository to keep it light. 

Download here (It's importatnt that you use the `cxx11 ABI` version, which works with OpenCV):

## Getting started:
```bash
wget https://download.pytorch.org/libtorch/cpu/libtorch-cxx11-abi-shared-with-deps-1.13.0%2Bcpu.zip
unzip libtorch-cxx11-abi-shared-with-deps-1.13.0+cpu.zip
```
rename the folder to 'libtorch' if it isn't already. 


Then, in the example-app, the first time you might have to run these commands. 
 `DCMAKE_PREFIX_PATH` is required to be an absolute path!
 
```bash
mkdir build
cd build
cmake -DCMAKE_PREFIX_PATH=/path/to/libtorch ..
cmake --build . --config Release
```
Note the `..` after the command! This references the parent directory.

For subsequent builds you can just type: 

## Run:
```
make
```

For first time install troubleshooting, see the [pytorch cppdocs](https://pytorch.org/cppdocs/installing.html), which this information is based on.
