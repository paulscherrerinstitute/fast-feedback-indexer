## Fast feedback indexer C++ library

This directory contains the code for the fast feedback indexer library, implementaed in C++17 and CUDA.

* The algorithm description is here: https://github.com/paulscherrerinstitute/fast-feedback-indexer/tree/main/doc/algorithm
* The API description is here: https://github.com/paulscherrerinstitute/fast-feedback-indexer/tree/main/doc/api

### Vector candidate refinement

The algorithm features four steps

 1. Vector candidate sampling (on GPU)
 1. Vector candidate refinement (on GPU, optional)
 1. Unit cell sampling (on GPU)
 1. Unit cell refinement (on CPU)

The vector candidate refinement can be switched off using the VECTOR_CANDIDATE_REFINEMENT cmake option.

### Performance issues

* Keep memory for each coordinate dimension separate to avoid a stride of 3 elements in GPU kernels
* Consider replacing sort + merge top in gpu_find_candidates with partitioning algorithms
* *unsigned* as data type for a block sequentializer in device is most likely inadequate, what is fastest?

### Dependencies

* C++17
* CUDA Runtime
* Eigen for cell refining

### Memory handling

* All memory is allocated once per indexer object on creation

### Memory pinning

Memory needs to be pinned and kept alive by the user.

### Multiple GPUs

Every indexer object can act on a separate GPU

* if *INDEXER_GPU_DEVICE* is set in the environment, the device is taken from that
* otherwise the current CUDA device is used

### Multiple GPU Streams

Every *fast_feedback::indexer* object uses a separate Cuda stream

### Documentation

* **TODO**: If required, use doxygen
* The main documentation is provided as LaTeX files in the doc folder

### Thread safety

* Threads can use their private *fast_feedback::indexer* object in parallel
* Methods on the indexer objects are not thread safe, except where stated otherwise (e.g. cell refinement on CPU)
* Logger is thread safe. Currently log output from different threads can get mingled (use LOG_START and LOG_END macros consistently to prevent that)

### Logging

Logging output steered by *INDEXER_LOG_LEVEL* goes to stdlog (the same as stderr), except logging output from the GPU device steered by *INDEXER_GPU_DEBUG*, which goes to stdout.

### Version

The library version (in the form of commit id and date) is printed as a log message at log levels info/debug.

### Environment Variables

Steer library behaviour with environment variables. 

* *INDEXER_LOG_LEVEL* (string): The log level for the indexer {"fatal", "error", "warn", "info", "debug"} (parsed on calling *logger::init_log_level()*)
* *INDEXER_GPU_DEVICE* (int): The GPU cuda device number to use for indexing (parsed on indexer object creation)
* *INDEXER_GPU_DEBUG* (string): Print gpu kernel debug output to stdout {"1", "true", "yes", "on", "0", "false", "no", "off"} (parsed on indexer object creation)
* *INDEXER_VERBOSE_EXCEPTION* (string): Add (file:line) before exception message {"1", "true", "yes", "on", "0", "false", "no", "off"} (parsed on exception creation)

### Noteworthy Cmake Variables

* CMAKE_CUDA_ARCHITECTURES: GPU architecture, default \"75;80\"
   * https://cmake.org/cmake/help/latest/variable/CMAKE_CUDA_ARCHITECTURES.html
   * https://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc/index.html#gpu-feature-list
* CMAKE_CUDA_FLAGS: Extra flags to pass to the cuda compiler
   * https://cmake.org/cmake/help/latest/envvar/CUDAFLAGS.html
* CXXFLAGS: Extra flags for initializing CMAKE_CXX_FLAGS
   * https://cmake.org/cmake/help/latest/variable/CMAKE_LANG_FLAGS.html
* CXX: C++ Compiler
   * https://cmake.org/cmake/help/latest/envvar/CXX.html#cxx

### Meson

The Meson build differs in certain points from the CMake build.

* GPU and CPU architecture can be left undefined
* Less control over rpaths and compiler options
* Option cpu-arch=xxx sets the -march=xxx compiler flag
* Option gpu-arch=xxx sets the --gencode=xxx nvcc flag
* CXX and CXXFLAGS will be picked up in the build setup
* NVCC_PREPEND_FLAGS and NVCC_APPEND_FLAGS can be used to set nvcc flags at compile time

See the meson.options file for available options.

### Using the library in your own code

Depending on what is included in the build, the setup-env.sh script sets up a bash environment that enables

* PKG_CONFIG_PATH for the Linux pkg-config mechanism ('fast_indexer' or 'fast_indexer_static') to pickup linking flags and include directories, e.g. in Cmake or meson projects
* LD_LIBRARY_PATH for linking at runtime
* CPLUS_INCLUDE_PATH and C_INCLUDE_PATH for picking up includes
* PYTHONPATH for importing the ffbidx python package
