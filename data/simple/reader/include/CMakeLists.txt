cmake_minimum_required (VERSION 3.0)
project(simple_data "CXX")

find_package (Eigen3 3.3 REQUIRED NO_MODULE)

add_library(simple_data INTERFACE)
target_include_directories(simple_data INTERFACE simple_data.h)

target_link_libraries(simple_data INTERFACE Eigen3::Eigen)

