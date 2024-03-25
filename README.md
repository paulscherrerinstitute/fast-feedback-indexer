# Fast feedback indexer

Develop an indexer for fast feedback

*Status*: We're optimistic that it does what it's supposed to. *Luis Barbas* PyTorch implementation has been benchmarked extensively here at PSI and shows indexing quality on par with other known indexers and superior speed. This CUDA version is behind the PyTorch implementation, but I'm trying to catch up.

*Issues*: Implemented in CUDA, so only Nvidia GPUs are supported currently. Good guess of initial cell required.

### Alternative Implementations

*Luis Barba* from the Swiss Data Science Center has done an implementation with PyTorch: https://renkulab.io/projects/lfbarba/toro-indexer-for-serial-crystallography

### Referencing

When referring to the algorithm implemented here, please use:
 
DOI: https://doi.org/10.26434/chemrxiv-2023-wnm9n

*TODO*: Add non-preprint version in the future

### Attributions

This is the result of the REDML Project, and other more or less informal collaborations between

* REDML: PSI MX Group (https://www.psi.ch/en/macromolecular-crystallography)
* REDML: Swiss Data Science Center (https://datascience.ch)
* REDML: PSI AWI (https://www.psi.ch/en/awi)
* REDML: CSCS (https://www.cscs.ch)
* Graeme Winter, Nick Devenish, Richard Gildea - DIALS group at Diamond Light Source (https://dials.github.io/about.html)

### External Build Dependencies

* C++17 compatible compiler
* cmake > 3.21 (not so sure if it works with earlier versions as well)
* Eigen3 header only library
* *BUILD_FAST_INDEXER* needs a compiler compatible CUDA toolkit
* *PYTHON_MODULE* needs Python3 and NumPy (also see https://cmake.org/cmake/help/latest/module/FindPython3.html)

### Internal Build Dependencies

Cmake should complain and tell you what to add to the cmake commandline if internal dependencies are not met. In general tests require what is tested, executables need the simple data reader and indexer libraries, and the python module needs the indexer library.

### Build Instructions

If the default value of *CMAKE_CUDA_ARCHITECTURES*=\"75;80\" is inappropriate for your GPU, set it correctly, or go to the bottom of the README file for the indexer to get more info.

Get hold of the Eigen3 library, either by installing it via your distro (e.g. on Ubuntu: `sudo apt install libeigen3-dev`) or download it using `git submodule update`. Install python3 and numpy for the python module.

```
$ FFBIDX_INSTALL_DIR=${HOME}/ffbidx
$ mkdir build
$ cd build
$ # For PSI merlin cluster add
$ # export Eigen3_DIR=<path to eigen installation>
$ cmake -DCMAKE_INSTALL_PREFIX=${FFBIDX_INSTALL_DIR} -DCMAKE_BUILD_TYPE=Release \
  -DINSTALL_SIMPLE_DATA_FILES=ON -DTEST_ALL=ON -DTESTS_RPATH=ON \
  -DBUILD_SIMPLE_DATA_READER=ON -DPYTHON_MODULE=ON \
  -DPYTHON_MODULE_RPATH=ON ..
$ make
$ ctest
```

### Installation Instructions

Take note of the CMAKE_INSTALL_PREFIX value above.

```
$ make install
$ # The following only works for sh compatible shells.
$ # Adapt the script if you're using something else.
$ . ${FFBIDX_INSTALL_DIR}/share/ffbidx/setup-env.sh
```

As a quick installation test you could do

```
$ refined_simple_data_indexer \
  ${FFBIDX_INSTALL_DIR}/share/ffbidx/data/files/image0_local.txt \
  300 1 1 $((32*1024)) false ifss .8 6 15
$ python -c "import ffbidx; print('OK')"
```

### Installation with Spack on custom systems

Install the official Spack instance
```
git clone https://github.com/spack/spack.git
source spack/share/spack/setup-env.sh
```

Tell Spack to find your C++17 compatible compiler
```
spack compiler find # HPC users should use 'module load gcc/<version>' beforehand
```

Add the unofficial ffbidx spack recipe
```
spack repo add fast-feedback-indexer/spack
```

Install ffbidx and its dependencies
```
spack install ffbidx cuda_arch=60 +python +simple_data_indexer +simple_data_files +test_all
# Not all options are mandatory
# Change your cuda_arch according to the system you are running on.
```

Before using the lib in C++ load its run env using:
```
spack load ffbidx
```

### Installation with Spack on Merlin (PSI cluster)

See: https://git.psi.ch/germann_e/spack-psi
