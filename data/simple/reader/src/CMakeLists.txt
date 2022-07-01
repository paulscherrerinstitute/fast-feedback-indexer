cmake_minimum_required (VERSION 3.0)
project(simple_data_reader_test "CXX")

find_package(simple_data REQUIRED NO_MODULE)

add_executable(test simple_data_test.cpp)

target_link_libraries(test INTERFACE simple_data)
add_test(NAME read_file_0 COMMAND test)
set_property(TEST read_file_0 PROPERTY PASS_REGULAR_EXPRESSION "^Test OK")
set_property(TEST read_file_0 PROPERTY FAIL_REGULAR_EXPRESSION "^Error")
