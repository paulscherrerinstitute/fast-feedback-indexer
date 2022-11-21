project(python_impl
        DESCRIPTION "Python module"
        LANGUAGES CXX)

option(PYTHON_MODULE "Enable python module" OFF)
option(PYTHON_MODULE_RPATH "Set python module RPATH to fast indexer library installation" OFF)
set(PYTHON_MODULE_PATH "${CMAKE_INSTALL_LIBDIR}/ffbidx" CACHE PATH "Indexer python module installation path")

if(PYTHON_MODULE)
        #find_package(CUDAToolkit REQUIRED)
        find_package(Python3 COMPONENTS NumPy Development.Module)
        if(NOT ${Python3_Development.Module_FOUND})
                message(FATAL_ERROR "Python3 module development components not found!")
        endif()
        if(NOT ${Python3_NumPy_FOUND})
                message(FATAL_ERROR "Python3 NumPy development components not found!")
        endif()
        if(NOT BUILD_FAST_INDEXER)
                message(FATAL_ERROR "PYTHON_MODULE needs -DBUILD_FAST_INDEXER=1 as a cmake argument")
        endif()
        Python3_add_library(ffbidx MODULE
                python_module.cpp)
        target_compile_features(ffbidx PRIVATE cxx_std_17)
        target_include_directories(ffbidx PUBLIC ${Python3_INCLUDE_DIRS} ${Python3_NumPy_INCLUDE_DIRS})
        target_link_libraries(ffbidx
                PRIVATE fast_indexer
                PUBLIC Python3::NumPy
                PUBLIC Python3::Module)
        if (PYTHON_MODULE_RPATH)
                set_target_properties(ffbidx PROPERTIES
                        INSTALL_RPATH ${CMAKE_INSTALL_LIBDIR})
        endif(PYTHON_MODULE_RPATH)
        install(FILES $<TARGET_FILE:ffbidx>
                DESTINATION ${PYTHON_MODULE_PATH}/ffbidx
                COMPONENT ffbidx_python
                RENAME __init__.so)
        cmake_path(ABSOLUTE_PATH PYTHON_MODULE_PATH
                BASE_DIRECTORY ${CMAKE_INSTALL_PREFIX}
                NORMALIZE
                OUTPUT_VARIABLE ffbidx_PYTHONPATH)
        install(CODE "MESSAGE(\"use PYTHONPATH=${ffbidx_PYTHONPATH}\")"
                COMPONENT ffbidx_python)
endif(PYTHON_MODULE)
