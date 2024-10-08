# ---- Indexer Tests ----

option(TEST_INDEXER_SIMPLE "Enable ctest test code for simple indexer test" OFF)
option(TEST_INDEXER_EXCEPTION "Enable ctest test code for indexer exception test" OFF)
option(TEST_INDEXER_OBJ "Enable ctest test code for indexer object code" OFF)
option(SIMPLE_DATA_INDEXER "Enable test executable for indexing of simple data" OFF)
option(REFINED_SIMPLE_DATA_INDEXER "Enable test executable for refined indexing of simple data" OFF)
option(TESTS_RPATH "Set RPATH for test executables to fast indexer library installation" OFF)

if(TEST_ALL)
        set(TEST_INDEXER ON)
endif(TEST_ALL)
                
if(TEST_INDEXER)
        set(TEST_INDEXER_SIMPLE ON)
        set(TEST_INDEXER_EXCEPTION ON)
        set(TEST_INDEXER_OBJ ON)
        set(SIMPLE_DATA_INDEXER ON)
        set(REFINED_SIMPLE_DATA_INDEXER ON)
endif(TEST_INDEXER)

if(TEST_INDEXER_SIMPLE)
        set(THREADS_PREFER_PTHREAD_FLAG ON)
        find_package(Threads REQUIRED)
        if(NOT BUILD_SIMPLE_DATA_READER)
                message(FATAL_ERROR "TEST_INDEXER_SIMPLE needs -DBUILD_SIMPLE_DATA_READER=1 as a cmake argument")
        endif()
        if(NOT BUILD_FAST_INDEXER)
                message(FATAL_ERROR "TEST_INDEXER_SIMPLE needs -DBUILD_FAST_INDEXER=1 as a cmake argument")
        endif()

        add_executable(test_indexer test_indexer.cpp)
        target_compile_features(test_indexer PRIVATE cxx_std_17)
        target_link_libraries(test_indexer
                PRIVATE fast_indexer
                PRIVATE simple_data
                PRIVATE Threads::Threads)
        add_test(NAME index_file_0 COMMAND test_indexer $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/data/simple/files/image0_local.txt>)
        set_property(TEST index_file_0 PROPERTY PASS_REGULAR_EXPRESSION "Test OK")
        set_property(TEST index_file_0 PROPERTY FAIL_REGULAR_EXPRESSION "Test failed")

        add_executable(test_capi test_capi.cpp)
        target_compile_features(test_capi PRIVATE cxx_std_17)
        target_link_libraries(test_capi
                PRIVATE fast_indexer
                PRIVATE simple_data)
        add_test(NAME capi_index_file_0 COMMAND test_capi $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/data/simple/files/image0_local.txt>)
        set_property(TEST capi_index_file_0 PROPERTY PASS_REGULAR_EXPRESSION "Test OK")
        set_property(TEST capi_index_file_0 PROPERTY FAIL_REGULAR_EXPRESSION "Test failed")
endif(TEST_INDEXER_SIMPLE)

if(TEST_INDEXER_EXCEPTION)
        if(NOT BUILD_FAST_INDEXER)
                message(FATAL_ERROR "TEST_INDEXER_SIMPLE needs -DBUILD_FAST_INDEXER=1 as a cmake argument")
        endif()
        add_executable(test_indexer_exception test_indexer_exception.cpp)
        target_compile_features(test_indexer_exception PRIVATE cxx_std_17)
        target_link_libraries(test_indexer_exception
                PRIVATE fast_indexer)
        add_test(NAME fast_feedback_exception COMMAND test_indexer_exception)
        set_property(TEST fast_feedback_exception PROPERTY PASS_REGULAR_EXPRESSION "Test OK")
        set_property(TEST fast_feedback_exception PROPERTY FAIL_REGULAR_EXPRESSION "Test failed")
endif(TEST_INDEXER_EXCEPTION)

if(TEST_INDEXER_OBJ)
        if(NOT BUILD_FAST_INDEXER)
                message(FATAL_ERROR "TEST_INDEXER_SIMPLE needs -DBUILD_FAST_INDEXER=1 as a cmake argument")
        endif()
        add_executable(test_indexer_obj test_indexer_obj.cpp)
        target_compile_features(test_indexer_obj PRIVATE cxx_std_17)
        target_link_libraries(test_indexer_obj
                PRIVATE fast_indexer)
        add_test(NAME indexer_object COMMAND test_indexer_obj)
        set_property(TEST indexer_object PROPERTY PASS_REGULAR_EXPRESSION "Test OK")
        set_property(TEST indexer_object PROPERTY FAIL_REGULAR_EXPRESSION "Test failed")
endif(TEST_INDEXER_OBJ)

if (TESTS_RPATH)
        cmake_path(ABSOLUTE_PATH CMAKE_INSTALL_LIBDIR
                BASE_DIRECTORY ${CMAKE_INSTALL_PREFIX}
                NORMALIZE
                OUTPUT_VARIABLE tests_RPATH)
        if (INSTALL_RELOCATABLE)
                cmake_path(ABSOLUTE_PATH CMAKE_INSTALL_BINDIR
                        BASE_DIRECTORY ${CMAKE_INSTALL_PREFIX}
                        NORMALIZE
                        OUTPUT_VARIABLE tests_ABS_PATH)
                cmake_path(RELATIVE_PATH tests_RPATH
                        BASE_DIRECTORY ${tests_ABS_PATH})
                set(tests_RPATH $ORIGIN/${tests_RPATH})
        endif(INSTALL_RELOCATABLE)
endif(TESTS_RPATH)

if(SIMPLE_DATA_INDEXER)
        if(NOT BUILD_SIMPLE_DATA_READER)
                message(FATAL_ERROR "SIMPLE_DATA_INDEXER needs -DBUILD_SIMPLE_DATA_READER=1 as a cmake argument")
        endif()
        if(NOT BUILD_FAST_INDEXER)
                message(FATAL_ERROR "SIMPLE_DATA_INDEXER needs -DBUILD_FAST_INDEXER=1 as a cmake argument")
        endif()
        add_executable(simple_data_indexer simple_data_indexer.cpp)
        target_compile_features(simple_data_indexer PRIVATE cxx_std_17)
        target_link_libraries(simple_data_indexer
                PRIVATE fast_indexer
                PRIVATE simple_data)
        if (TESTS_RPATH)
                set_target_properties(simple_data_indexer PROPERTIES
                        INSTALL_RPATH ${tests_RPATH})
        endif(TESTS_RPATH)
        install(TARGETS simple_data_indexer
                RUNTIME
                DESTINATION ${CMAKE_INSTALL_BINDIR}
                COMPONENT ffbidx_executables)
endif(SIMPLE_DATA_INDEXER)

if(REFINED_SIMPLE_DATA_INDEXER)
        if(NOT BUILD_SIMPLE_DATA_READER)
                message(FATAL_ERROR "SIMPLE_DATA_INDEXER needs -DBUILD_SIMPLE_DATA_READER=1 as a cmake argument")
        endif()
        if(NOT BUILD_FAST_INDEXER)
                message(FATAL_ERROR "SIMPLE_DATA_INDEXER needs -DBUILD_FAST_INDEXER=1 as a cmake argument")
        endif()
        add_executable(refined_simple_data_indexer refined_simple_data_indexer.cpp)
        target_compile_features(refined_simple_data_indexer PRIVATE cxx_std_17)
        target_link_libraries(refined_simple_data_indexer
                PRIVATE fast_indexer
                PRIVATE simple_data)
        if (TESTS_RPATH)
                set_target_properties(refined_simple_data_indexer PROPERTIES
                        INSTALL_RPATH ${tests_RPATH})
        endif(TESTS_RPATH)
        install(TARGETS refined_simple_data_indexer
                RUNTIME
                DESTINATION ${CMAKE_INSTALL_BINDIR}
                COMPONENT ffbidx_executables)
endif(REFINED_SIMPLE_DATA_INDEXER)

if (SIMPLE_DATA_INDEXER OR REFINED_SIMPLE_DATA_INDEXER)
        cmake_path(ABSOLUTE_PATH CMAKE_INSTALL_BINDIR
                BASE_DIRECTORY ${CMAKE_INSTALL_PREFIX}
                NORMALIZE
                OUTPUT_VARIABLE ffbidx_PATH)
        install(CODE "MESSAGE(\"use PATH=${ffbidx_PATH}\")"
                COMPONENT ffbidx_executables)
        string(APPEND fast_feedback_indexer_ENV_SETUP "ffbidx_PATH=${ffbidx_PATH};")
        set(fast_feedback_indexer_ENV_SETUP "${fast_feedback_indexer_ENV_SETUP}" CACHE INTERNAL ENV)
endif()

# ---- Data Tests ----

option(TEST_DATA "Add targets for the data tests" OFF)
IF(TEST_ALL)
        set(TEST_DATA ON)
ENDIF(TEST_ALL)

option(TEST_SIMPLE_DATA "Add targets for testing simple data" OFF)
IF(TEST_DATA)
        set(TEST_SIMPLE_DATA ON)
ENDIF(TEST_DATA)

option(TEST_SIMPLE_DATA_READER "Enable ctest test code for simple data reader" OFF)
if (TEST_SIMPLE_DATA)
            set(TEST_SIMPLE_DATA_READER ON)
endif(TEST_SIMPLE_DATA)

if(TEST_SIMPLE_DATA_READER)
        if(NOT BUILD_SIMPLE_DATA_READER)
                message(FATAL_ERROR "TEST_INDEXER_SIMPLE needs -DBUILD_SIMPLE_DATA_READER=1 as a cmake argument")
        endif()
        add_executable(test_simple_data_reader simple_data_test.cpp)
        target_link_libraries(test_simple_data_reader
                PRIVATE simple_data)
        add_test(NAME read_file_0 COMMAND test_simple_data_reader $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/data/simple/files/image0_local.txt>)
        set_property(TEST read_file_0 PROPERTY PASS_REGULAR_EXPRESSION "^Test OK")
        set_property(TEST read_file_0 PROPERTY FAIL_REGULAR_EXPRESSION "^Error")
endif(TEST_SIMPLE_DATA_READER)
