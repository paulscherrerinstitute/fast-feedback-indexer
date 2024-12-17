find_package(Git)

if(GIT_EXECUTABLE)
    execute_process(
        COMMAND ${GIT_EXECUTABLE} log --pretty=format:%h -1
        WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
        OUTPUT_VARIABLE GIT_VERSION_TXT
        RESULT_VARIABLE ERROR_CODE
        OUTPUT_STRIP_TRAILING_WHITESPACE
    )
endif()

if(GIT_VERSION_TXT STREQUAL "")
  set(VERSION_TXT "\"${VRS}\"")
else()
  set(VERSION_TXT "\"${VRS}-${GIT_VERSION_TXT}\"")
endif()

message("Version detection in directory ${CMAKE_SOURCE_DIR} found: ${VERSION_TXT}")

configure_file(${SRC} ${DST} @ONLY)
