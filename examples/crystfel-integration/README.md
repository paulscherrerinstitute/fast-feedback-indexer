# CrystFEL integration

1. Compile FFBIDX with additional option to include CrystFEL integration:

```
mkdir -p build
cd build
/usr/bin/cmake .. -DBUILD_CRYSTFEL_INTEGRATION=ON -DCMAKE_INSTALL_PREFIX=<directory>
make
sudo make install
```

This will place FFBIDX files in <directory> (if it is not included, /usr/local is the target directory).

2. Compile CrystFEL

Requires [special version](https://github.com/fleon-psi/crystfel/tree/fast_indexer) from CrystFEL at the moment.

```
git clone https://github.com/fleon-psi/crystfel
git checkout fast_indexer
mkdir -p build
cd build
/usr/bin/cmake .. -DCMAKE_INSTALL_PREFIX=<directory>
make
```

PKG_CONFIG_PATH may be necessary to detect fast feedback indexer.
