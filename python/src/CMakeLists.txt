option(PYTHON_MODULE "Enable python module" OFF)
option(PYTHON_MODULE_RPATH "Set python module RPATH to fast indexer library installation" OFF)
set(PYTHON_MODULE_PATH "${CMAKE_INSTALL_LIBDIR}" CACHE PATH "Indexer python module installation path")

if(PYTHON_MODULE)
        find_package(Eigen3 3 NO_MODULE)
        if(NOT ${Eigen3_FOUND})
                message(FATAL_ERROR
                        "Eigen3 library not found! Install a distro specific package, or use 'git submodule update' to download the submodule.")
        endif()
        find_package(Python3 COMPONENTS NumPy Development.Module Development.Embed)
        if(NOT ${Python3_Development.Module_FOUND})
                message(FATAL_ERROR "Python3 module development components not found!")
        endif()
        if(NOT ${Python3_NumPy_FOUND})
                message(FATAL_ERROR "Python3 NumPy development components not found!")
        endif()
        if (NOT ${Python3_Development.Embed_FOUND})
                message(FATAL_ERROR "Python3 Embed development components not found!")
        endif()
        message(STATUS "Python: found ${Python3_INCLUDE_DIRS}")
        message(STATUS "NumPy: found ${Python3_NumPy_INCLUDE_DIRS}")
        if(NOT BUILD_FAST_INDEXER)
                message(FATAL_ERROR "PYTHON_MODULE needs -DBUILD_FAST_INDEXER=1 as a cmake argument")
        endif()
        Python3_add_library(ffbidx SHARED
                ffbidx_impl.cpp)
        target_compile_features(ffbidx PRIVATE cxx_std_17)
        target_include_directories(ffbidx PUBLIC ${Python3_INCLUDE_DIRS} ${Python3_NumPy_INCLUDE_DIRS})
        target_link_libraries(ffbidx
                PRIVATE fast_indexer
                PUBLIC Python3::NumPy
                PUBLIC Python3::Module
                INTERFACE Eigen3::Eigen)
        set(ffbidx_INSTALL_PATH ${PYTHON_MODULE_PATH}/ffbidx)
        set(ffbidx_INSTALL_RPATH $ORIGIN)
        if (PYTHON_MODULE_RPATH)
                cmake_path(ABSOLUTE_PATH CMAKE_INSTALL_LIBDIR
                        BASE_DIRECTORY ${CMAKE_INSTALL_PREFIX}
                        NORMALIZE
                        OUTPUT_VARIABLE ffbidx_RPATH)
                if (INSTALL_RELOCATABLE)
                        cmake_path(ABSOLUTE_PATH ffbidx_INSTALL_PATH
                                BASE_DIRECTORY ${CMAKE_INSTALL_PREFIX}
                                NORMALIZE
                                OUTPUT_VARIABLE ffbidx_ABS_PATH)
                        cmake_path(RELATIVE_PATH ffbidx_RPATH
                                BASE_DIRECTORY ${ffbidx_ABS_PATH})
                        set(ffbidx_RPATH $ORIGIN/${ffbidx_RPATH})
                endif(INSTALL_RELOCATABLE)
                set(ffbidx_INSTALL_RPATH ${ffbidx_INSTALL_RPATH}:${ffbidx_RPATH}:${Python3_RUNTIME_LIBRARY_DIRS})
        endif(PYTHON_MODULE_RPATH)
        set_target_properties(ffbidx PROPERTIES
                PREFIX ""
                OUTPUT_NAME "ffbidx_impl"
                INSTALL_RPATH ${ffbidx_INSTALL_RPATH})
        install(TARGETS ffbidx
                DESTINATION ${ffbidx_INSTALL_PATH}
                COMPONENT ffbidx_python)
        install(FILES __init__.py ffbidx_cl.py
                DESTINATION ${ffbidx_INSTALL_PATH}
                COMPONENT ffbidx_python)
        cmake_path(ABSOLUTE_PATH PYTHON_MODULE_PATH
                BASE_DIRECTORY ${CMAKE_INSTALL_PREFIX}
                NORMALIZE
                OUTPUT_VARIABLE ffbidx_PYTHONPATH)
        install(CODE "MESSAGE(\"use PYTHONPATH=${ffbidx_PYTHONPATH}\")"
                COMPONENT ffbidx_python)
        string(APPEND fast_feedback_indexer_ENV_SETUP "ffbidx_PYTHONPATH=${ffbidx_PYTHONPATH};")
        set(fast_feedback_indexer_ENV_SETUP "${fast_feedback_indexer_ENV_SETUP}" CACHE INTERNAL ENV)
endif(PYTHON_MODULE)
