# Libtorch and Opencv

A minimal example of how ot use OpenCV and Libtorch in the same project.

The project expects libtorch/ in the top-level direcotry. I have not included this in the repository to keep it light. 

Download here (cxx11 ABI):

```bash
wget https://download.pytorch.org/libtorch/cpu/libtorch-cxx11-abi-shared-with-deps-1.13.0%2Bcpu.zip
unzip libtorch-cxx11-abi-shared-with-deps-1.13.0+cpu.zip
```

Then, in the example-app, the first time you might have to run these commands. 
 `DCMAKE_PREFIX_PATH` is required to be an absolute path!
 
```bash
mkdir build
cd build
cmake -DCMAKE_PREFIX_PATH=/path/to/libtorch ..
cmake --build . --config Release
```


For subsequent builds, you can use the Makefile for convenience.
For first time install, check this https://pytorch.org/cppdocs/installing.html

