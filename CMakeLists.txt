cmake_minimum_required(VERSION 3.14)
project(Feature_Extractor)

set(CMAKE_CXX_STANDARD 14)

find_package(OpenCV 3.0 QUIET)
find_package(glog)
include_directories(${OpenCV_INCLUDE_DIRS})

add_executable(distribute_oct_tree src/oct_tree/main.cpp)
target_link_libraries(distribute_oct_tree ${OpenCV_LIBS})

add_executable(ORB_SLAM_Extractor src/orb_extractor/main_orb_extractor.cpp src/orb_extractor/ORBExtractor.cpp)
target_link_libraries(ORB_SLAM_Extractor ${OpenCV_LIBS} glog::glog)