cmake_minimum_required (VERSION 3.0)
project(data)

option(TEST_DATA "Add targets for the data tests" OFF)
IF(TEST_ALL)
    set(TEST_DATA ON)
ENDIF(TEST_ALL)

add_subdirectory(simple)
