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

### Environment Variables

Steer program startup arguments with environment variables. If required, cli args parsing can be introduced as well.

* *INDEXER_GPU_DEVICE* (int): The GPU cuda device number to use for indexing
* *INDEXER_LOG_LEVEL* (string): The log level for the indexer {"fatal", "error", "warn", "info", "debug"}
