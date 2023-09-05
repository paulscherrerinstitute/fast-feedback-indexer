# CrystFEL integration

1. Compile FFBIDX with additional option to include CrystFEL integration:

```
export BASE_DIR=<directory>

git clone https://github.com/paulscherrerinstitute/fast-feedback-indexer/
cd fast-feedback-indexer
git submodule update --init
mkdir build
cd build
cmake .. -DCMAKE_CUDA_ARCHITECTURES="70;75;80" -DBUILD_CRYSTFEL_INTEGRATION=ON -DCMAKE_INSTALL_PREFIX=${BASE_DIR}/ffbidx
make
sudo make install
```

This will place FFBIDX files in \<directory\>/ffbidx (if it is not included, /usr/local is the target directory).
The CUDA architecture settings are optimized for V100/A100/T4 GPU devices. You might want to change CUDA architectures of you have older/newer devices, see [Nvidia documentation](https://developer.nvidia.com/cuda-gpus) for details.

2. Compile CrystFEL with the following options:
```
export BASE_DIR=<directory>
export PKG_CONFIG_PATH=$PKG_CONFIG_PATH:${BASE_DIR}/ffbidx/lib/pkgconfig
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:${BASE_DIR}/ffbidx/lib

git clone https://github.com/fleon-psi/crystfel
cd crystfel
git checkout fast_indexer
mkdir -p build
cd build
cmake .. -DCMAKE_INSTALL_PREFIX=${BASE_DIR}/crystfel-ffbidx
make
make install

```

Requires [forked version](https://github.com/fleon-psi/crystfel/tree/fast_indexer) of CrystFEL at the moment.
\<directory\> has to be the same as used for FFBIDX.

Then you can add "--indexing=ffbidx" to indexamajig options. FFBIDX lib directory has to be in the LD_LIBRARY_PATH for the execution.

To print possible configuration options, you can run:
indexamajig --help-ffbidx
