function usage() {
    echo "usage: $0 [version]"
    echo
    echo "Replace version strings with new version"
    exit -1
}

if (($# == 0)); then
    grep VERSION meson.build
elif (($# == 1)); then
    echo "replace version strings by $1"
    sed -i 's/VERSION\(.*\)\w\+\.\w\+\.\w\+/VERSION\1'"$1"'/' CMakeLists.txt
    sed -i 's/VERSION_TXT\(.*\)\w\+\.\w\+\.\w\+/VERSION_TXT\1'"$1"'/' GenerateVersionH.cmake
    sed -i 's/VERSION\(.*\)\w\+\.\w\+\.\w\+/VERSION\1'"$1"'/' meson.build
    sed -i 's/version_txt\(.*\)\w\+\.\w\+\.\w\+/version_txt\1'"$1"'/' ffbidx/meson.build
else
    usage
fi
