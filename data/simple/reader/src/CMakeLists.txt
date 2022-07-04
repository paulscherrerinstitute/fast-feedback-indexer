cmake_minimum_required (VERSION 3.0)
project(test_simple_data "CXX")

find_package(Eigen3 3.3 REQUIRED NO_MODULE)

add_executable(test_simple_data_reader simple_data_test.cpp)
target_include_directories(test_simple_data_reader PUBLIC ${EIGEN3_INCLUDE_DIR})
target_link_libraries(test_simple_data_reader INTERFACE Eigen3::Eigen)

add_test(NAME read_file_0 COMMAND test_simple_data_reader $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/data/simple/image0_local.txt>)
set_property(TEST read_file_0 PROPERTY PASS_REGULAR_EXPRESSION "^Test OK")
set_property(TEST read_file_0 PROPERTY FAIL_REGULAR_EXPRESSION "^Error")
