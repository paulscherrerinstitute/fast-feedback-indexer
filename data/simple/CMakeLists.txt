cmake_minimum_required (VERSION 3.0)
project(simple_data)

option(TEST_SIMPLE_DATA "Add targets for testing simple data" OFF)
IF(TEST_DATA)
    set(TEST_SIMPLE_DATA ON)
ENDIF(TEST_DATA)

add_subdirectory(reader)
