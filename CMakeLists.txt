cmake_minimum_required (VERSION 3.21)
project(fast_feedback_indexer)

include(GNUInstallDirs)
include(CMakePackageConfigHelpers)

option(TEST_ALL "Add test targets" OFF)
option(INSTALL_RELOCATABLE "Relocatable installation (especially RPATH)" ON)

enable_testing()

set(fast_feedback_indexer_ENV_SETUP "" CACHE INTERNAL ENV)

set(SOURCE_VERSION_FILE "${CMAKE_CURRENT_BINARY_DIR}/source_version.txt")
set(SOURCE_UNKNOWN_VERSION "unknown:non-git-version")
if (EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/.git")
    find_package(Git)
    if (GIT_FOUND)
        add_custom_command(
            COMMAND ${GIT_EXECUTABLE} describe --tags --always > "${SOURCE_VERSION_FILE}"
            WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
            OUTPUT "${SOURCE_VERSION_FILE}"
            VERBATIM)
    else()
        file(WRITE "${SOURCE_VERSION_FILE}" "${SOURCE_UNKNOWN_VERSION}")
    endif()
else()
    file(WRITE "${SOURCE_VERSION_FILE}" "${SOURCE_UNKNOWN_VERSION}")
endif()
add_custom_target(source_version ALL
  DEPENDS ${SOURCE_VERSION_FILE})

if (EXISTS ${CMAKE_SOURCE_DIR}/eigen/CMakeLists.txt)
    add_subdirectory(eigen)
endif()
add_subdirectory(indexer)
add_subdirectory(data)
add_subdirectory(tests)
add_subdirectory(python)
add_subdirectory(examples/crystfel-integration)

MESSAGE("ENV ${fast_feedback_indexer_ENV_SETUP}")
configure_file(setup-env.sh.in
    setup-env.sh
    @ONLY)
set(ffbidx_ENV_PATH ${CMAKE_INSTALL_DATADIR}/ffbidx)
install(FILES ${CMAKE_CURRENT_BINARY_DIR}/setup-env.sh
    DESTINATION ${ffbidx_ENV_PATH}
    COMPONENT ffbidx_common)
cmake_path(ABSOLUTE_PATH ffbidx_ENV_PATH
    BASE_DIRECTORY ${CMAKE_INSTALL_PREFIX}
    NORMALIZE
    OUTPUT_VARIABLE ffbidx_ENV_SCRIPT_PATH)
install(CODE "MESSAGE(\"for sh compatible shells use
    . ${ffbidx_ENV_SCRIPT_PATH}/setup-env.sh\")")
