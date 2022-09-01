# Fast feedback indexer

Develop an indexer for fast feedback

### Build Dependencies

* C++17 compatible compiler
* indexer needs a compiler compatible CUDA toolkit
* data/simple/reader needs the Eigen 3.3 library

### Build Instructions

Get hold of the Eigen3 library, either by installing it via your distro (e.g. on Ubuntu: `sudo apt install libeigen3-dev`) or download it using `git submodule update`.

```
> mkdir build
> cd build
> cmake -DTEST_ALL=1 -DBUILD_SIMPLE_DATA_READER=1 ..
> make
> ctest
```

### Installation Instructions

TODO
