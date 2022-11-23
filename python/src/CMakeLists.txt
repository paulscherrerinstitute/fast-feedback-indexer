project(python_impl
        DESCRIPTION "Python module"
        LANGUAGES CXX)

option(PYTHON_MODULE "Enable python module" OFF)
option(PYTHON_MODULE_RPATH "Set python module RPATH to fast indexer library installation" OFF)
set(PYTHON_MODULE_PATH "${CMAKE_INSTALL_LIBDIR}/ffbidx" CACHE PATH "Indexer python module installation path")

if(PYTHON_MODULE)
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
        if(NOT BUILD_FAST_INDEXER)
                message(FATAL_ERROR "PYTHON_MODULE needs -DBUILD_FAST_INDEXER=1 as a cmake argument")
        endif()
        Python3_add_library(ffbidx SHARED
                python_module.cpp)
        target_compile_features(ffbidx PRIVATE cxx_std_17)
        target_include_directories(ffbidx PUBLIC ${Python3_INCLUDE_DIRS} ${Python3_NumPy_INCLUDE_DIRS})
        target_link_libraries(ffbidx
                PRIVATE fast_indexer
                PUBLIC Python3::NumPy
                PUBLIC Python3::Module)
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
                set(ffbidx_INSTALL_RPATH ${ffbidx_INSTALL_RPATH}:${ffbidx_RPATH})
        endif(PYTHON_MODULE_RPATH)
        set_target_properties(ffbidx PROPERTIES
                PREFIX ""
                OUTPUT_NAME "__init__"
                INSTALL_RPATH ${ffbidx_INSTALL_RPATH})
        install(TARGETS ffbidx
                DESTINATION ${ffbidx_INSTALL_PATH}
                COMPONENT ffbidx_python)
        cmake_path(ABSOLUTE_PATH PYTHON_MODULE_PATH
                BASE_DIRECTORY ${CMAKE_INSTALL_PREFIX}
                NORMALIZE
                OUTPUT_VARIABLE ffbidx_PYTHONPATH)
        install(CODE "MESSAGE(\"use PYTHONPATH=${ffbidx_PYTHONPATH}\")"
                COMPONENT ffbidx_python)
endif(PYTHON_MODULE)
