cmake_minimum_required(VERSION 3.0 FATAL_ERROR) 
project(example-app)
list(APPEND CMAKE_PREFIX_PATH "/home/cyrill/pytorch/libtorch")


find_package(Torch REQUIRED)
FIND_PACKAGE( OpenCV REQUIRED COMPONENTS core imgproc highgui)                              
INCLUDE_DIRECTORIES(${OpenCV_INCLUDE_DIRS})
SET(GCC_COVERAGE_COMPILE_FLAGS "-fprofile-arcs -ftest-coverage")



add_executable(example-app example-app.cpp)
target_link_libraries(example-app ${TORCH_LIBRARIES}; ${OpenCV_LIBS})
set_property(TARGET example-app PROPERTY CXX_STANDARD 14)
