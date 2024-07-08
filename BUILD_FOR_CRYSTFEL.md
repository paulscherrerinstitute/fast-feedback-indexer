# Building the Fast Feedback Indexer Library for CrystFEL on Linux Systems

Since CrystFEL uses meson as it's build system, the following will assume meson is installed. The code must be compiled with a C++17 and gnu compiler option compatible compiler (like g++-13 or clang++-18). A CUDA installation is also required. The code requires the Eigen library.

The file meson.options contains default values for meson compilation and build options. Such an option can also be set on the commandline with -D option='xxx'.

## Eigen Library (Option eigen-source-dir)

If you have a working installation of the Eigen library, this option can be set to 'ignore'. Otherwise you can set it to 'eigen' after checking out the Eigen library source code with

    $ git submodule update --force --recursive

If you manage your own Eigen installation, set the option to the directory that contains the 'Eigen' subdirectory with the header files.

## Collect System Information

The library build uses CPU and GPU architecture specific optimization flags. In order to determine these flags, it's necessary to know what CPU and GPU hardware the library code needs to run on. 

### Determine GPU Compute Capability (Option gpu-arch)

In order to determin the compute capability of the GPUs the code should be able to run on, use the nvidia-smi command.

    $ nvidia-smi --query-gpu=compute_cap --format=csv,noheader

This information will end up as a single --gencode='xxx' nvidia compiler option, see [nvcc compiler options](https://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc/index.html#options-for-steering-gpu-code-generation). So if the output of the nvidia-smi comand is 8.9, the value of 'xxx' can be set to 'arch=compute_89,code=sm_89'

The value can be set to 'ignore', then no --gencode option will be given to the Nvidia compiler.

### Determine CPU Architecture (Option cpu-arch)

This is a bit tricky, as it requires some knowledge of CPUs. The CPU architecture option 'xxx' will end up in a -march='xxx' g++ compiler option (also valid for clang++), see [gcc cpu architecture options](https://gcc.gnu.org/onlinedocs/gcc/x86-Options.html). There are three ways for setting this.

#### Set to 'ignore'

No -march option will be given to the compiler.

#### Set to 'native'

The value will be set according to the CPU architecture the code is compiled on. If your compilation machine is never than the machines you run the library code on, that will most likely fail with 'illegal instruction' crashes. Make sure you compile the code on the machine with the oldest CPU, that is likely to work, at least for AMD and Intel CPUs.

#### Set to specific architecture

The following command will show the CPU model

    $ lscpu

Then the model needs to mapped to the architecture, e.g. with the help of [Wikipedias list of AMD processors](https://en.wikipedia.org/wiki/List_of_AMD_processors) or similar.

## Install

Choose an installation destination, here ${PWD}/install, and once the meson.options file has the desired options, compile and install the library with

    $ INSTALL_DIR=${PWD}/install
    $ CXX=g++-13 meson setup --reconfigure --buildtype=release --prefix=${INSTALL_DIR} --libdir=lib meson
    $ cd meson
    $ meson compile
    $ meson install

To subsequently use the library, setup an sh compatible shell environment with

    $ source ${INSTALL_DIR}/share/ffbidx/setup-env.sh

Amongst others, this sets up the LIBRARY_PATH and PKG_CONFIG_PATH.
