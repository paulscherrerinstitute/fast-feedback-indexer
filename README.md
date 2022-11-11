# Fast feedback indexer

Develop an indexer for fast feedback

*Status*: Current results based on simple brute force sampling seem to make sense.

### Build Dependencies

* C++17 compatible compiler
* indexer needs a compiler compatible CUDA toolkit
* data/simple/reader needs the Eigen 3.3 library
* python module needs Python3 and NumPy (also see https://cmake.org/cmake/help/latest/module/FindPython3.html)

### Build Instructions

Get hold of the Eigen3 library, either by installing it via your distro (e.g. on Ubuntu: `sudo apt install libeigen3-dev`) or download it using `git submodule update`.

```
> mkdir build
> cd build
> cmake -DTEST_ALL=1 -DBUILD_SIMPLE_DATA_READER=1 ..
> make
> ctest
```

For depositing the python module in /tmp/test on a Linux system use

```
> cd build
> cmake ... -DPYTHON_MODULE=1 ..
> make
> cd ..
> sh pythonlib.sh install
```

### Installation Instructions

TODO

*make install* doesn't work
