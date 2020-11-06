cmake_minimum_required(VERSION 3.14)
project(ExtractORB)

set(CMAKE_CXX_STANDARD 11)

set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS}  -Wall   -O3")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall   -O3")

find_package(OpenCV 3)
MESSAGE("OPENCV VERSION:")
MESSAGE(${OpenCV_VERSION})
find_package(Eigen3 3)
find_package(glog)
find_package(Pangolin REQUIRED)
include_directories(
        ${OpenCV_INCLUDE_DIRS}
        ${EIGEN3_INCLUDE_DIR}
        ${CMAKE_SOURCE_DIR}
        ${CMAKE_SOURCE_DIR}/inc
        ${CMAKE_SOURCE_DIR}/inc/CameraModels
)

add_executable(distribute_oct_tree src/oct_tree/main.cpp)
target_link_libraries(distribute_oct_tree ${OpenCV_LIBS} glog::glog ${EIGEN3_LIBS})

add_executable(ORB_SLAM_Extractor src/orb_extractor/main_orb_extractor.cpp src/orb_extractor/ORBExtractor.cpp)
target_link_libraries(ORB_SLAM_Extractor ${OpenCV_LIBS} glog::glog ${EIGEN3_LIBS})

add_executable(clahe_img_keypoint
        src/clahe/main_show_clahe_keypoint.cpp
        src/orb_extractor/ORBExtractor.cpp)
target_link_libraries(clahe_img_keypoint ${OpenCV_LIBS} glog::glog ${EIGEN3_LIBS})

add_executable(whole_extractor
        src/orb_extractor/main_whole_orb_extractor.cpp
        src/orb_extractor/ORBextractor.cc
        src/Frame.cc
        src/ImuTypes.cc
        src/Converter.cc
        src/G2oTypes.cc
        src/MapPoint.cc
        src/Map.cc
        src/ORBmatcher.cc
        src/KeyFrame.cc
        src/CameraModels/KannalaBrandt8.cpp
        src/CameraModels/Pinhole.cpp
        src/TwoViewReconstruction.cc
        src/KeyFrameDatabase.cc
        )
target_link_libraries(whole_extractor
        ${OpenCV_LIBS} glog::glog ${EIGEN3_LIBS} ${Pangolin_LIBRARIES}
        ${PROJECT_SOURCE_DIR}/Thirdparty/DBoW2/lib/libDBoW2.so
        ${PROJECT_SOURCE_DIR}/Thirdparty/g2o/lib/libg2o.so
        )

add_executable(
        clahe
        src/clahe/main_clahe.cpp
)
target_link_libraries(
        clahe
        ${OpenCV_LIBS} glog::glog
)

add_executable(
        frame
        src/matcher/main_frame.cpp
        src/orb_extractor/ORBextractor.cc
        src/Frame.cc
        src/ImuTypes.cc
        src/Converter.cc
        src/G2oTypes.cc
        src/MapPoint.cc
        src/Map.cc
        src/ORBmatcher.cc
        src/KeyFrame.cc
        src/CameraModels/KannalaBrandt8.cpp
        src/CameraModels/Pinhole.cpp
        src/TwoViewReconstruction.cc
        src/KeyFrameDatabase.cc
)
target_link_libraries(
        frame
        ${OpenCV_LIBS} glog::glog ${EIGEN3_LIBS} ${Pangolin_LIBRARIES}
        ${PROJECT_SOURCE_DIR}/Thirdparty/DBoW2/lib/libDBoW2.so
        ${PROJECT_SOURCE_DIR}/Thirdparty/g2o/lib/libg2o.so
)

#add_executable(
#        undistort
#        src/undistort/main_undistort.cpp
#        src/orb_extractor/ORBExtractor.cpp
#        src/undistort/Frame_changed.cpp
#        src/CameraModels/KannalaBrandt8.cpp
#        src/CameraModels/Pinhole.cpp
#        src/undistort/TwoViewReconstruction.cc
#)
#target_link_libraries(undistort ${OpenCV_LIBS} glog::glog ${EIGEN3_LIBS}
#        ${PROJECT_SOURCE_DIR}/Thirdparty/DBoW2/lib/libDBoW2.so
#        ${PROJECT_SOURCE_DIR}/Thirdparty/g2o/lib/libg2o.so
#        )

add_executable(
        matcher
        src/matcher/main_matcher.cpp
        src/orb_extractor/ORBextractor.cc
        src/Frame.cc
        src/ImuTypes.cc
        src/Converter.cc
        src/G2oTypes.cc
        src/MapPoint.cc
        src/Map.cc
        src/ORBmatcher.cc
        src/KeyFrame.cc
        src/CameraModels/KannalaBrandt8.cpp
        src/CameraModels/Pinhole.cpp
        src/TwoViewReconstruction.cc
        src/KeyFrameDatabase.cc
)
target_link_libraries(
        matcher
        ${OpenCV_LIBS} glog::glog ${EIGEN3_LIBS} ${Pangolin_LIBRARIES}
        ${PROJECT_SOURCE_DIR}/Thirdparty/DBoW2/lib/libDBoW2.so
        ${PROJECT_SOURCE_DIR}/Thirdparty/g2o/lib/libg2o.so
)