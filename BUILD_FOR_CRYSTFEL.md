# Building the Fast Feedback Indexer Library for CrystFEL on Linux Systems

Since CrystFEL uses meson as it's build system, the following will assume meson is installed. The code must be compiled with a C++17 and gnu compiler option compatible compiler (like g++-13 or clang++-18). A CUDA installation is also required. The code requires the Eigen library.

## Install the Fast Feedback Indexer Library

The file meson.options contains default values for meson compilation and build options. Such an option can also be set on the commandline with -D option='xxx'.

Start with cloning the fast feedback indexer library repository:

    $ git clone https://github.com/paulscherrerinstitute/fast-feedback-indexer
    $ cd fast-feedback-indexer

### Eigen Library (Option eigen-source-dir)

If you have a working installation of the Eigen library, this option can be set to 'ignore'. Otherwise you can set it to 'eigen' after checking out the Eigen library source code with

    $ git submodule update --force --recursive

If you manage your own Eigen installation, set the option to the directory that contains the 'Eigen' subdirectory with the header files.

### Collect System Information

The library build uses CPU and GPU architecture specific optimization flags. In order to determine these flags, it's necessary to know what CPU and GPU hardware the library code needs to run on. 

#### Determine GPU Compute Capability (Option gpu-arch)

In order to determin the compute capability of the GPUs the code should be able to run on, use the nvidia-smi command.

    $ nvidia-smi --query-gpu=compute_cap --format=csv,noheader

This information will end up as a single --gencode='xxx' nvidia compiler option, see [nvcc compiler options](https://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc/index.html#options-for-steering-gpu-code-generation). So if the output of the nvidia-smi comand is 8.9, the value of 'xxx' can be set to 'arch=compute_89,code=sm_89'

The value can be set to 'ignore', then no --gencode option will be given to the Nvidia compiler.

#### Determine CPU Architecture (Option cpu-arch)

This is a bit tricky, as it requires some knowledge of CPUs. The CPU architecture option 'xxx' will end up in a -march='xxx' g++ compiler option (also valid for clang++), see [gcc cpu architecture options](https://gcc.gnu.org/onlinedocs/gcc/x86-Options.html). There are three ways for setting this.

##### Set to 'ignore'

No -march option will be given to the compiler.

##### Set to 'native'

The value will be set according to the CPU architecture the code is compiled on. If your compilation machine is never than the machines you run the library code on, that will most likely fail with 'illegal instruction' crashes. Make sure you compile the code on the machine with the oldest CPU - that is likely to work, at least for AMD and Intel CPUs.

##### Set to specific architecture

The following command will show the CPU model

    $ lscpu

Then the model needs to mapped to the architecture, e.g. with the help of [Wikipedias list of AMD processors](https://en.wikipedia.org/wiki/List_of_AMD_processors) or similar.

### Install

Choose an installation destination, here ${BASE_DIR}/ffbidx, and once the meson.options file has the desired options, compile and install the library with

    $ INSTALL_DIR=${BASE_DIR}/ffbidx
    $ CXX=g++-13 meson setup --reconfigure --buildtype=release --prefix=${INSTALL_DIR} --libdir=lib build-ffbidx
    $ cd build-ffbidx
    $ meson compile
    $ meson install

### Use

To subsequently use the library, setup an sh compatible shell environment with

    $ source ${INSTALL_DIR}/share/ffbidx/setup-env.sh

Amongst others, this sets up the LD_LIBRARY_PATH and PKG_CONFIG_PATH. The pkg-config files will be located at ${INSTALL_DIR}/share/ffbidx/pkgconfig.

## Build and Install CrystFEL

Presently, a special branch of a [forked version](https://github.com/fleon-psi/crystfel/tree/fast_indexer) of CrystFEL by *Filip Leonarski* is required to integrate the fast feedback indexer into CrystFEL. Clone the code with

    git clone -b fast_indexer_new_c_api https://github.com/fleon-psi/crystfel
    cd crystfel

Configure the build with

    meson setup -Dprefix=${BASE_DIR}/crystfel build-crystfel

Then compile and install the CrystFEL code with

    cd build-crystfel
    meson compile
    meson install

Then you can add "--indexing=ffbidx" to indexamajig options. Fast Feedback Indexer and CrystFEL lib directories have to be in the LD_LIBRARY_PATH for the execution, for the Fast Feedback Indexer this is easiest achieved by setting up the sh compatible shell environment as described above under **USE**.

To print possible configuration options, you can run:
indexamajig --help-ffbidx
