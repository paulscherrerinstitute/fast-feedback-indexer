### Performance issues

* Keep memory for each coordinate dimension separate to avoid a stride of 3 elements in GPU kernels
* Consider replacing sort + merge top in gpu_find_candidates with partitioning algorithms
* unsigned as data type for a block sequentializer in device is most likely inadequate, what is fastest?

### Dependencies

* Minimize indexer API include dependencies, try to rely on standard C++17 only
* Minimize linkage dependencies, try to rely on standard C++17 and CUDA runtime only

### Memory handling

* All memory should be allocated once at startup

### Memory pinning

* Who does memory pinning for asynchronous memory transfers from/to GPU?
* Currently the client is responsible for pinning the memory, maybe a second option would be convenient.

### Multiple GPUs

* I think it's most efficient to use multiple GPUs for different indexing problems. This sometimes increases latency, but maximizes throughput.
* But if required, multiple GPUs could collaborate on the same indexing problem.
* Multi GPU collaboration is necessary if the data doesn't fit onto the GPU, which I think is not a danger for indexing.

### Multiple GPU Streams

* Currently only stream 0 is used
* **TODO**: make every indexer instance use its own stream != 0

### Documentation

* **TODO**: If required, use doxygen
* Developper info in markdown files
* **TODO**: Maybe add user documentation in a separate doc folder with markdown files

### Thread safety

* Threads should be able to use their private indexer in parallel
* Logger must be thread safe. Currently log output from different threads can get mingled (if required implement per thread log cache, e.g. ostringstream, with coordinated flushing to final destination)

### Logging

Logging output steered by *INDEXER_LOG_LEVEL* goes to stdlog (the same as stderr), except logging output from the GPU device steered by *INDEXER_GPU_DEBUG*, which goes to stdout.

### Environment Variables

Steer program startup arguments with environment variables. If required, cli args parsing can be introduced as well.

* *INDEXER_GPU_DEVICE* (int): The GPU cuda device number to use for indexing
* *INDEXER_LOG_LEVEL* (string): The log level for the indexer {"fatal", "error", "warn", "info", "debug"}
* *INDEXER_GPU_DEBUG* (string): Print gpu kernel debug output to stdout {"1", "true", "yes", "on", "0", "false", "no", "off"}

### Noteworthy Cmake Variables

* CMAKE_CUDA_ARCHITECTURES: GPU architecture, default \"75;80\"
   * https://cmake.org/cmake/help/latest/variable/CMAKE_CUDA_ARCHITECTURES.html
   * https://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc/index.html#gpu-feature-list

