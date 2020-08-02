cmake_minimum_required(VERSION 3.14)
project(distribute_oct_tree)

set(CMAKE_CXX_STANDARD 14)

find_package(OpenCV 3.0 QUIET)
include_directories(${OpenCV_INCLUDE_DIRS})

add_executable(distribute_oct_tree main.cpp)
target_link_libraries(distribute_oct_tree ${OpenCV_LIBS})

