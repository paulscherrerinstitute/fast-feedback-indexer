CXX=g++ meson setup --reconfigure --buildtype=release --prefix=${PREFIX} --libdir=lib -Dinclude-python-api=enabled -Deigen-source-dir=eigen meson
cd meson
meson compile
meson install
mkdir -p ${SP_DIR}
mv ${PREFIX}/lib/ffbidx ${SP_DIR}
