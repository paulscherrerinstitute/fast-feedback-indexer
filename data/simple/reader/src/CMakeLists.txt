cmake_minimum_required (VERSION 3.0)
project(simple_data "CXX")

find_package(Eigen3 3.3 REQUIRED NO_MODULE)

add_executable(simple_data_reader_test simple_data_test.cpp)
target_include_directories(simple_data_reader_test PUBLIC ${EIGEN3_INCLUDE_DIR})
target_link_libraries(simple_data_reader_test INTERFACE Eigen3::Eigen)

add_test(NAME read_file_0 COMMAND simple_data_reader_test $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/data/simple/image0_local.txt>)
set_property(TEST read_file_0 PROPERTY PASS_REGULAR_EXPRESSION "^Test OK")
set_property(TEST read_file_0 PROPERTY FAIL_REGULAR_EXPRESSION "^Error")
