cmake_minimum_required(VERSION 3.0 FATAL_ERROR) 
project(example-app)

find_package(Torch REQUIRED)
FIND_PACKAGE( OpenCV REQUIRED COMPONENTS core imgproc highgui)
INCLUDE_DIRECTORIES(${OpenCV_INCLUDE_DIRS})


# profiling
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -O0")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O0")

set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -O3")

message(STATUS "CXX_FLAGS = ${CMAKE_CXX_FLAGS_RELEASE}")

add_executable(main main.cpp
        resize_longest_size.cpp
        resize_longest_size.h
        predictor.cpp
        predictor.h
        yolo.cpp
        yolo.h
)

target_link_libraries(main ${TORCH_LIBRARIES}; ${OpenCV_LIBS})

set_property(TARGET main PROPERTY CXX_STANDARD 17)
