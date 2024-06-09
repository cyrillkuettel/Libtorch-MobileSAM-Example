cmake_minimum_required(VERSION 3.10)
project(example-app)

set(CMAKE_SYSTEM_PROCESSOR arm64)
set(CMAKE_OSX_ARCHITECTURES arm64)

# Set C++ 17 compiler flags
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Profiling
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -O0")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O0")
set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -O3")

# Torch and OpenCV
find_package(Torch REQUIRED)
find_package(OpenCV REQUIRED COMPONENTS core imgproc highgui)

set(
        example_app_sources_list
        "mobile_sam.cpp"
        "mobile_sam.hpp"
        "resize_longest_size.cpp"
        "resize_longest_size.h"
        "predictor.cpp"
        "predictor.h"
        "yolo.cpp"
        "yolo.h"
        "visualize.cpp"
        "visualize.hpp"
)

# Create library target
add_library(${PROJECT_NAME} ${example_app_sources_list})
target_link_libraries(${PROJECT_NAME} ${TORCH_LIBRARIES} ${OpenCV_LIBS})
target_include_directories(${PROJECT_NAME} PRIVATE ${TORCH_INCLUDE_DIRS})
target_compile_definitions(${PROJECT_NAME} PRIVATE USE_HEADER_ONLY_LIB)

# Create executable target for main.cpp
add_executable(example_main example_main.cpp)
target_link_libraries(example_main ${PROJECT_NAME})

# Test sources
set(
        unittests_source_list
        "tests/catch.hpp"
        "tests/Main.cpp"
        "tests/UnitTest.cpp"
        "tests/UnitTest.h"
        "tests/PredictorTest.cpp"
)

# Build UnitTests executable
add_executable(UnitTests ${unittests_source_list})
target_link_libraries(UnitTests ${PROJECT_NAME})
set_property(TARGET UnitTests PROPERTY VS_STARTUP_PROJECT UnitTests)
