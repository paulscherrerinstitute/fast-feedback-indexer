#!/bin/sh

# Install or remove the python package from the build dir.
# This script only works if executed from the top directory of the repository.
#
# Usage: sh pythonlib.sh install|remove <package destination dir>
#        The default package destination dir is /tmp/ffbidx
# env var: CMAKE_BUILD_DIR the build dir for cmake
# Result: python lib in the package destination dir, or removed package dir

default_lib_dir=/tmp/test/ffbidx
default_build_dir=./build
lib_name=__init__.so

install() {
    local build_dir=${CMAKE_BUILD_DIR:-${default_build_dir}}
    echo "looking for build artefacts in ${build_dir}"
    local lib=${build_dir}/python/src/${lib_name}
    if [ ! -x "${lib}" ]; then
        echo "Failed: no library at ${lib}"
        echo "        build the library and set CMAKE_BUILD_DIR to the build directory"
        exit 1
    fi
    local dir="${1:-${default_lib_dir}}"
    local dest="${dir}/__init__.so"
    echo "install to ${dir}"
    mkdir -p "${dir}" && cp "${lib}" "${dest}" && echo "done, use PYTHONPATH=$(dirname "${dir}")"
}

remove() {
    local dir="${1:-${default_lib_dir}}"
    echo "remove ${dir}"
    if [ -d "${dir}" ]; then
        rm -rf "${dir}" && echo "done"
    fi
}

if test -r pythonlib.sh; then
    if [ "$1" = "install" ]; then
        install "$2"
    elif [ "$1" = "remove" ]; then
        remove "$2"
    else
        echo "Failed: use $0 install|remove [ <python lib dir, default={default_lib_dir} ]"
    fi
else
    echo "Failed: execute this command from the top directory of the repository!"
fi
