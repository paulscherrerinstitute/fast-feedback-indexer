find_package(Git)

if(GIT_EXECUTABLE)
    execute_process(
        COMMAND ${GIT_EXECUTABLE} log --pretty=format:"%H: %aD" -1
        WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
        OUTPUT_VARIABLE VERSION_TXT
        RESULT_VARIABLE ERROR_CODE
        OUTPUT_STRIP_TRAILING_WHITESPACE
    )
endif()

if(VERSION_TXT STREQUAL "")
  set(VERSION_TXT "${PROJECT_VERSION}")
  message(WARNING "Failed to determine version from Git.")
endif()

message("Version detection in directory ${CMAKE_SOURCE_DIR} found: ${VERSION_TXT}")

configure_file(${SRC} ${DST} @ONLY)
