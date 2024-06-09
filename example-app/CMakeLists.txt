cmake_minimum_required (VERSION 3.10)

# Set project name
project(example-app)

# For mac m2?
# set(CMAKE_SYSTEM_PROCESSOR arm64)
# set(CMAKE_OSX_ARCHITECTURES arm64)

# Build a test or utility executable
function(build_executable project_name source_list is_linked is_startup requires_pthreads)

    # Create executable as project name
    add_executable(${project_name} ${source_list})

    # Set C++ 17 compiler flags
    set(CMAKE_CXX_STANDARD 17)
    set(CMAKE_CXX_STANDARD_REQUIRED ON)
    set(CMAKE_CXX_EXTENSIONS OFF)

    target_link_libraries(${project_name} ${TORCH_LIBRARIES}; ${OpenCV_LIBS})

    # Set the executable as the default startup project
    if (is_startup)
        set_property(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR} PROPERTY VS_STARTUP_PROJECT ${project_name})
    endif ()

    target_compile_definitions(${project_name} PRIVATE USE_HEADER_ONLY_LIB)

    # Create IDE folder structure
    source_group("Source" FILES ${source_list})


endfunction()


# profiling
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -O0")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O0")
set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -O3")


# Set C++ 17 compiler flags
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)


# Torch and OpenCV
find_package(Torch REQUIRED)
FIND_PACKAGE(OpenCV REQUIRED COMPONENTS core imgproc highgui)
INCLUDE_DIRECTORIES(${OpenCV_INCLUDE_DIRS})

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
message(STATUS "CXX_FLAGS = ${CMAKE_CXX_FLAGS_RELEASE}")
set_property(TARGET ${PROJECT_NAME} PROPERTY CXX_STANDARD 17)


# Create additionally executable target for main.cpp
add_executable(example_main example_main.cpp)
target_link_libraries(example_main ${PROJECT_NAME} ${TORCH_LIBRARIES} ${OpenCV_LIBS})
target_include_directories(example_main PRIVATE ${TORCH_INCLUDE_DIRS})


# test sources
set(
        unittests_source_list
        "tests/catch.hpp"
        "tests/Main.cpp"
        #add more:
        "tests/UnitTest.cpp"
        "tests/UnitTest.h"
        "tests/MyUnitTestCode.cpp"
)

build_executable(UnitTests "${unittests_source_list}" TRUE TRUE FALSE)
target_include_directories(UnitTests PRIVATE ${TORCH_INCLUDE_DIRS})
target_link_libraries(UnitTests ${TORCH_LIBRARIES} ${OpenCV_LIBS})
