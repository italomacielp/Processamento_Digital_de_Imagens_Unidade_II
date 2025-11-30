cmake_minimum_required(VERSION 3.0.0)
project(threshold VERSION 0.1.0 LANGUAGES C CXX)

find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

add_executable(threshold threshold.cpp)
target_link_libraries(threshold ${OpenCV_LIBS})

add_executable(threshold2 threshold2.cpp)
target_link_libraries(threshold2 ${OpenCV_LIBS})
