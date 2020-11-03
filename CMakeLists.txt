cmake_minimum_required(VERSION 3.14)
project(ExtractORB)

set(CMAKE_CXX_STANDARD 14)

find_package(OpenCV 3)
find_package(Eigen3 3)
find_package(glog)
include_directories(
        ${OpenCV_INCLUDE_DIRS}
        ${EIGEN3_INCLUDE_DIR}
        ${CMAKE_SOURCE_DIR}/inc
        ${CMAKE_SOURCE_DIR}/inc/CameraModels
)

add_executable(distribute_oct_tree src/oct_tree/main.cpp)
target_link_libraries(distribute_oct_tree ${OpenCV_LIBS} glog::glog ${EIGEN3_LIBS})

add_executable(ORB_SLAM_Extractor src/orb_extractor/main_orb_extractor.cpp src/orb_extractor/ORBExtractor.cpp)
target_link_libraries(ORB_SLAM_Extractor ${OpenCV_LIBS} glog::glog ${EIGEN3_LIBS})

add_executable(undistort src/undistort/main_undistort.cpp src/orb_extractor/ORBExtractor.cpp src/undistort/Frame.cpp src/CameraModels/KannalaBrandt8.cpp src/CameraModels/Pinhole.cpp)
target_link_libraries(undistort ${OpenCV_LIBS} glog::glog ${EIGEN3_LIBS})