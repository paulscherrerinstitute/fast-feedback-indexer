# Building the Fast Feedback Indexer Library for CrystFEL on Linux Systems

Since CrystFEL uses meson as it's build system, the following will assume meson is installed. The code must be compiled with a C++17 and gnu compiler option compatible compiler (like g++-13 or clang++-18). A CUDA installation, compatible with your compiler and GPU driver, is also required. The code requires the Eigen library.

## Install the Fast Feedback Indexer Library

The file meson.options contains default values for meson compilation and build options. Such an option can also be set on the commandline with -D option='xxx'.

Start with cloning the fast feedback indexer library repository:

    $ git clone https://github.com/paulscherrerinstitute/fast-feedback-indexer
    $ cd fast-feedback-indexer

Then you can use the newest code as is, or check out a release, e.g. v1.1.2, with

   $ git checkout v1.1.2

### Eigen Library (Option eigen-source-dir)

If you have a working installation of the Eigen library, this option can be set to 'ignore'. Otherwise you can set it to 'eigen' after checking out the Eigen library source code with

    $ git submodule update --force --recursive --init

If you manage your own Eigen installation, set the option to the directory that contains the 'Eigen' subdirectory with the header files.

### Collect System Information

The library build uses CPU and GPU architecture specific optimization flags. In order to determine these flags, it's necessary to know what CPU and GPU hardware the library code needs to run on. 

#### Determine GPU Compute Capability (Option gpu-arch)

In order to determin the compute capability of the GPUs the code should be able to run on, use the nvidia-smi command.

    $ nvidia-smi --query-gpu=compute_cap --format=csv,noheader

This information will end up as a single --gencode='xxx' nvidia compiler option, see [nvcc compiler options](https://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc/index.html#options-for-steering-gpu-code-generation). So if the output of the nvidia-smi comand is 8.9, the value of 'xxx' can be set to 'arch=compute_89,code=sm_89'. The value can be set to 'ignore', then no --gencode option will be given to the Nvidia compiler.

If you optimize the code this way, it's important to be aware of the fact that the code will only be able to run on the devices compiled for. The value can be set to 'ignore', then no --gencode option will be given to the Nvidia compiler.

In general, the Nvidia GPU driver must be compatible with the output produced by the nvcc CUDA compiler. The version of the driver and compiler can be printed with

    $ nvidia-smi --query-gpu=driver_version --format=csv,noheader
    $ nvcc --version

If the driver is not compatible (e.g. too old), you'll have to ask the system administrator to upgrade the driver. Alternatively, you may downgrade your CUDA installation (of which nvcc is a part of) to a version compatible with the driver, see [CUDA compatibility](https://docs.nvidia.com/deploy/cuda-compatibility).

#### Determine CPU Architecture (Option cpu-arch)

This is a bit tricky, as it requires some knowledge of CPUs. The CPU architecture option 'xxx' will end up in a -march='xxx' g++ compiler option (also valid for clang++), see [gcc cpu architecture options](https://gcc.gnu.org/onlinedocs/gcc/x86-Options.html). There are three ways for setting this.

##### Set to 'ignore'

No -march option will be given to the compiler.

##### Set to 'native'

The value will be set according to the CPU architecture the code is compiled on. If your compilation machine is never than the machines you run the library code on, that will most likely fail with 'illegal instruction' crashes. Make sure you compile the code on the machine with the oldest CPU - that is likely to work, at least for AMD and Intel CPUs.

##### Set to specific architecture

The following command will show the CPU model

    $ lscpu

Then the model needs to be mapped to the architecture, e.g. with the help of [Wikipedias list of AMD processors](https://en.wikipedia.org/wiki/List_of_AMD_processors) or similar.

### Install

Choose an installation destination, here ${BASE_DIR}/ffbidx, and once the meson.options file has the desired options, compile and install the library with

    $ INSTALL_DIR=${BASE_DIR}/ffbidx
    $ meson setup --reconfigure --buildtype=release --prefix=${INSTALL_DIR} --libdir=lib build-ffbidx
    $ cd build-ffbidx
    $ meson compile
    $ meson install

### Use

To subsequently use the library, setup an sh compatible shell environment with

    $ source ${INSTALL_DIR}/share/ffbidx/setup-env.sh

Amongst others, this sets up the LD_LIBRARY_PATH and PKG_CONFIG_PATH. The pkg-config files will be located at ${INSTALL_DIR}/share/ffbidx/pkgconfig.

## Build and Install CrystFEL

Clone the code with

    git clone https://github.com/taw10/crystfel.git
    cd crystfel

Configure the build with

    meson setup --reconfigure --buildtype=release --prefix=${BASE_DIR}/crystfel build-crystfel

Make sure the runtime dependency 'fast_indexer' is found by checking the output. Then compile and install the CrystFEL code with

    cd build-crystfel
    meson compile
    meson install

Make sure '${BASE_DIR}/crystfel/bin' is in your PATH and '${BASE_DIR}/crystfel/lib/x86_64-linux-gnu' in your LD_LIBRARY_PATH.
Then you can add '--indexing=ffbidx' to indexamajig options to use the Fast Feedback Indexer.

Fast Feedback Indexer and CrystFEL lib directories have to be in the LD_LIBRARY_PATH for the execution, for the Fast Feedback Indexer this is easiest achieved by setting up the sh compatible shell environment as described above under **USE**.

To print possible configuration options for the Fast Feedback Indexer, you can run

    $ indexamajig --help-ffbidx

The indexer algorithm and its options are described in the LaTeX documents in the 'doc' folder of the Fast Feedback Indexer repository.

