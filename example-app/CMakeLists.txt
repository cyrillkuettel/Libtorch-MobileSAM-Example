cmake_minimum_required(VERSION 3.0 FATAL_ERROR)
project(example-app)


find_package(Torch REQUIRED)
FIND_PACKAGE( OpenCV REQUIRED COMPONENTS core imgproc highgui)                              
INCLUDE_DIRECTORIES( ${OpenCV_INCLUDE_DIRS} )
SET(GCC_OPENCV_FLAGS "-I/usr/local/include/opencv4 -L/usr/local/lib -lopencv_gapi -lopencv_stitching -lopencv_aruco -lopencv_barcode -lopencv_bgsegm -lopencv_bioinspired -lopencv_ccalib -lopencv_dnn_objdetect -lopencv_dnn_superres -lopencv_dpm -lopencv_face -lopencv_freetype -lopencv_fuzzy -lopencv_hfs -lopencv_img_hash -lopencv_intensity_transform -lopencv_line_descriptor -lopencv_mcc -lopencv_quality -lopencv_rapid -lopencv_reg -lopencv_rgbd -lopencv_saliency -lopencv_stereo -lopencv_structured_light -lopencv_phase_unwrapping -lopencv_superres -lopencv_optflow -lopencv_surface_matching -lopencv_tracking -lopencv_highgui -lopencv_datasets -lopencv_text -lopencv_plot -lopencv_videostab -lopencv_videoio -lopencv_wechat_qrcode -lopencv_xfeatures2d -lopencv_shape -lopencv_ml -lopencv_ximgproc -lopencv_video -lopencv_xobjdetect -lopencv_objdetect -lopencv_calib3d -lopencv_imgcodecs -lopencv_features2d -lopencv_dnn -lopencv_flann -lopencv_xphoto -lopencv_photo -lopencv_imgproc -lopencv_core")



set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS} ${GCC_OPENCV_FLAGS}")


add_executable(example-app example-app.cpp)
target_link_libraries(example-app "${TORCH_LIBRARIES}" "${OpenCV_LIBS}")
set_property(TARGET example-app PROPERTY CXX_STANDARD 14)
