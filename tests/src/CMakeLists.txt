project(tests_impl
        DESCRIPTION "Tests"
        LANGUAGES CXX CUDA)

# ---- Indexer Tests ----

option(TEST_INDEXER "Enable test code for indexer" OFF)
IF(TEST_ALL)
        set(TEST_INDEXER ON)
ENDIF(TEST_ALL)
                
option(TEST_INDEXER_SIMPLE "Enable test code for simple indexer test" OFF)
if (TEST_INDEXER)
        set(TEST_INDEXER_SIMPLE ON)
endif(TEST_INDEXER)

if(TEST_INDEXER_SIMPLE)
        set(BUILD_FAST_INDEXER ON)
        add_executable(test_indexer test_indexer.cpp)
        target_compile_features(test_indexer PRIVATE cxx_std_17)
        target_link_libraries(test_indexer
                PRIVATE fast_indexer
                PRIVATE simple_data)

        add_test(NAME index_file_0 COMMAND test_indexer $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/data/simple/files/image0_local.txt>)
        set_property(TEST index_file_0 PROPERTY PASS_REGULAR_EXPRESSION "Test OK")
        set_property(TEST index_file_0 PROPERTY FAIL_REGULAR_EXPRESSION "Test failed")
endif(TEST_INDEXER_SIMPLE)

# ---- Data Tests ----

option(TEST_DATA "Add targets for the data tests" OFF)
IF(TEST_ALL)
    set(TEST_DATA ON)
ENDIF(TEST_ALL)

option(TEST_SIMPLE_DATA "Add targets for testing simple data" OFF)
IF(TEST_DATA)
    set(TEST_SIMPLE_DATA ON)
ENDIF(TEST_DATA)

option(TEST_SIMPLE_DATA_READER "Enable test code for simple data reader" OFF)
if (TEST_SIMPLE_DATA)
    set(TEST_SIMPLE_DATA_READER ON)
endif(TEST_SIMPLE_DATA)

if(TEST_SIMPLE_DATA_READER)
    add_executable(test_simple_data_reader simple_data_test.cpp)
    target_link_libraries(test_simple_data_reader
        PRIVATE simple_data)

    add_test(NAME read_file_0 COMMAND test_simple_data_reader $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/data/simple/files/image0_local.txt>)
    set_property(TEST read_file_0 PROPERTY PASS_REGULAR_EXPRESSION "^Test OK")
    set_property(TEST read_file_0 PROPERTY FAIL_REGULAR_EXPRESSION "^Error")
endif(TEST_SIMPLE_DATA_READER)
