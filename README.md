# Fast feedback indexer

Develop an indexer for fast feedback

*Status*: Current results based on simple brute force sampling seem to make sense.

### External Build Dependencies

* C++17 compatible compiler
* cmake > 3.21 (not so sure if it works with earlier versions as well)
* indexer needs a compiler compatible CUDA toolkit
* *BUILD_SIMPLE_DATA_READER* and *REFINED_SIMPLE_DATA_INDEXER* need the Eigen 3.3 library
* *PYTHON_MODULE* needs Python3 and NumPy (also see https://cmake.org/cmake/help/latest/module/FindPython3.html)

### Internal Build Dependencies

Cmake should complain and tell you what to add to the cmake commandline if internal dependencies are not met. In general tests require what is tested, executables need the simple data reader and indexer libraries, and the python module needs the indexer library.

### Build Instructions

Get hold of the Eigen3 library, either by installing it via your distro (e.g. on Ubuntu: `sudo apt install libeigen3-dev`) or download it using `git submodule update`. Install python3 and numpy for the python module.

```
$ mkdir build
$ cd build
$ cmake -DCMAKE_INSTALL_PREFIX=${HOME}/ffbidx -DCMAKE_BUILD_TYPE=Release \
  -DINSTALL_SIMPLE_DATA_FILES=ON -DTEST_ALL=ON \
  -DBUILD_SIMPLE_DATA_READER=ON -DPYTHON_MODULE=ON ..
$ make
$ ctest
```

### Installation Instructions

Take note of the CMAKE_INSTALL_PREFIX value above.

```
$ make install
$ export LD_LIBRARY_PATH=${HOME}/ffbidx/lib
$ export PYTHONPATH=${HOME}/ffbidx/lib/ffbidx
```

As a quick installation test you could do

```
$ ${HOME}/ffbidx/bin/refined_simple_data_indexer ${HOME}/ffbidx/share/ffbidx/data/files/image0_local.txt 300 1 8 $((32*1024)) .2 .1
$ python -c "import ffbidx; print('OK')"
```
