# Fast feedback indexer

Develop an indexer for fast feedback

*Status*: We're optimistic that it does what it's supposed to. *Filip Leonarski* will integrate this code into CrystFEL.

*Luis Barbas* PyTorch implementation has been benchmarked extensively here at PSI and shows indexing quality on par with other known indexers and superior speed. This CUDA version has been tested internally by *Duan Jiaxin*, shows indexing quality on par with XGandalf and superior speed.

*Restrictions*: Initial cell geometry is required. In other words - this indexer cannot be used for finding a cell from scratch.

*Issues*: Implemented in CUDA, so only Nvidia GPUs are supported currently. With a suitable development machine, porting the code to HIP looks like a very realistic possibility. Porting to SYCL could also be an option.

### Alternative Implementations

*Luis Barba* from the Swiss Data Science Center has done an implementation with PyTorch: https://renkulab.io/projects/lfbarba/toro-indexer-for-serial-crystallography

The CUDA implementation exploits parallelism within the algorithm to a larger extent than the PyTorch version. That's why the PyTorch version relies on batching for speed, where the CUDA version can be fast taking frame by frame separately. The Python code of the PyTorch version is easier to understand and is thus recommended for experimentation with the algorithm.

### Referencing

When referring to the algorithm implemented here, please use:

- DOI: https://doi.org/10.1107/S1600576724003182
- *TODO*: Add some future paper with quality tests for the CUDA version

A copy of the BibTeX file can be found in this repo with the name *BIBTeX.bib*.

### Attributions

This is the result of the REDML Project, and other more or less informal collaborations between

* REDML: *Filip Leonarski*, *Duan Jiaxin* - PSI MX Group (https://www.psi.ch/en/macromolecular-crystallography)
* REDML: *Luis Barba*, *Benjamin Béjar* - Swiss Data Science Center (https://datascience.ch)
* REDML: *Piero Gasparotto* (formerly), *Hans-Christian Stadler*, *Greta Assmann*, *Elsa Germann* - PSI AWI (https://www.psi.ch/en/awi)
* REDML: *Henrique Mendonça* - CSCS (https://www.cscs.ch)
* Graeme Winter, Nick Devenish, Richard Gildea - DIALS group at Diamond Light Source (https://dials.github.io/about.html)
* David Waterman - CCP4 (https://www.ccp4.ac.uk)
* Thomas White - DESY (https://www.desy.de/~twhite/crystfel)

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
  300 1 1 $((32*1024)) false ifssr .8 .02 6 15
$ python -c "import ffbidx; print('OK')"
```

### Installation with Spack on custom systems

Due to *Elsa Germann*, we are able to provide spack integration.

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

### Installation with Meson
(This was tested with meson-1.4.0 and ninja-1.10.1)

In the top code directory do the following after adapting these lines to your own needs:
```
$ FFBIDX_INSTALL_DIR=${HOME}/ffbidx
$ CXX=g++-13 meson setup --reconfigure --buildtype=release --prefix=${FFBIDX_INSTALL_DIR} \
  --libdir=lib -D install-simple-data-reader=enabled -D install-simple-data-files=enabled \
  -D build-tests=enabled -D build-simple-indexers=enabled -D include-python-api=enabled \
  -D default_library=both -D gpu-arch='arch=compute_89,code=sm_89' meson
$ cd meson
$ meson compile -v
$ meson test
$ meson install
$ . ${FFBIDX_INSTALL_DIR}/share/ffbidx/setup-env.sh
```
Then do the qiuck installation test above.

### Installation for Python using micromamba and conda-build (experimental)
```
$ micromamba install conda-build
$ conda-build -c conda-forge conda/meta.yaml
$ micromamba install -c local ffbidx
$ python -c "import ffbidx; print('OK')"
```

### Installation for Python through conda-forge
TODO
