cmake_minimum_required(VERSION 3.0 FATAL_ERROR) 
project(example-app)

find_package(Torch REQUIRED)
FIND_PACKAGE( OpenCV REQUIRED COMPONENTS core imgproc highgui)
INCLUDE_DIRECTORIES(${OpenCV_INCLUDE_DIRS})

# Check if the compiler is GCC
if (CMAKE_COMPILER_IS_GNUCXX)
    set(GCC_COVERAGE_COMPILE_FLAGS "-fprofile-arcs -ftest-coverage")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${GCC_COVERAGE_COMPILE_FLAGS}")
endif()


set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -O3")

message(STATUS "CXX_FLAGS = ${CMAKE_CXX_FLAGS_RELEASE}")

add_executable(example-app example-app.cpp
        resize_longest_size.cpp
        resize_longest_size.h
        predictor.cpp
        predictor.h)
target_link_libraries(example-app ${TORCH_LIBRARIES}; ${OpenCV_LIBS})

set_property(TARGET example-app PROPERTY CXX_STANDARD 17) 


# add_executable(main main_sam.cpp)
# target_link_libraries(main ${TORCH_LIBRARIES}; ${OpenCV_LIBS})
# set_property(TARGET main PROPERTY CXX_STANDARD 14)
