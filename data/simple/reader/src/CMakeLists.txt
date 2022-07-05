cmake_minimum_required (VERSION 3.0)
cmake_policy(SET CMP0076 NEW)

project(simple_data_reader CXX)

find_package(Eigen3 3.3 REQUIRED NO_MODULE)

add_library(simple_data INTERFACE)
#target_compile_features(simple_data INTERFACE cxx_std_17)
target_sources(simple_data INTERFACE simple_data.h)
target_include_directories(simple_data INTERFACE .)
target_link_libraries(simple_data INTERFACE Eigen3::Eigen)
