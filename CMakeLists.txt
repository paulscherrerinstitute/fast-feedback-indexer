cmake_minimum_required (VERSION 3.0)
project(fast_feedback_indexer)

option(TEST_ALL "Add test targets" OFF)

enable_testing()
if (EXISTS ${CMAKE_SOURCE_DIR}/eigen/CMakeLists.txt)
    add_subdirectory(eigen)
endif()
add_subdirectory(indexer)
add_subdirectory(data)
add_subdirectory(tests)
add_subdirectory(python)
