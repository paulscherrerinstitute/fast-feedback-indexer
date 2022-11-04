project(python_impl
        DESCRIPTION "Python module"
        LANGUAGES CXX)

option(PYTHON_MODULE "Enable python module" OFF)

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
endif(PYTHON_MODULE)
