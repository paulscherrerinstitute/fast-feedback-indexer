## Fast feedback indexer C++ library

This directory contains the code for the fast feedback indexer library, implementaed in C++17 and CUDA.

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
* Eigen for lsq refined indexer

### Memory handling

* All memory is allocated once per indexer object on creation

### Memory pinning

* Who does memory pinning for asynchronous memory transfers from/to GPU?
* Currently the client is responsible for pinning the memory, maybe a second option would be convenient.

### Multiple GPUs

* Every indexer object can act on a separate GPU
   * if *INDEXER_GPU_DEVICE* is set in the environment, the device is taken from that
   * otherwise the current CUDA device is used
* I think it's most efficient to use multiple GPUs for different indexing problems. This sometimes increases latency, but maximizes throughput.
* But if required, multiple GPUs could collaborate on the same indexing problem.
* Multi GPU collaboration is necessary if the data doesn't fit onto the GPU, which I think is not a danger for indexing.

### Multiple GPU Streams

Every *fast_feedback::indexer* object uses a separate Cuda stream

### Documentation

* **TODO**: If required, use doxygen
* Developper info in markdown files
* **TODO**: Maybe add user documentation in a separate doc folder with markdown files

### Thread safety

* Threads should be able to use their private *fast_feedback::indexer* object in parallel
* Logger is thread safe. Currently log output from different threads can get mingled (use LOG_START and LOG_END macros consistently to prevent that)

### Logging

Logging output steered by *INDEXER_LOG_LEVEL* goes to stdlog (the same as stderr), except logging output from the GPU device steered by *INDEXER_GPU_DEBUG*, which goes to stdout.

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
