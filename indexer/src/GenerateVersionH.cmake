find_package(Git)

if(GIT_EXECUTABLE)
    # cmake_path(GET SRC ROOT_PATH WORKING_DIR)
    execute_process(
        COMMAND ${GIT_EXECUTABLE} log --pretty=format:"%H: %aD" -1
        # WORKING_DIRECTORY ${WORKING_DIR}
        OUTPUT_VARIABLE VERSION_TXT
        RESULT_VARIABLE ERROR_CODE
        OUTPUT_STRIP_TRAILING_WHITESPACE
    )
endif()

if(VERSION_TXT STREQUAL "")
  set(VERSION_TXT unknown)
  message(WARNING "Failed to determine version from Git.")
endif()

configure_file(${SRC} ${DST} @ONLY)
