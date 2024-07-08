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
    sed -i 's/VERSION\(.*\)\w\.\w\.\w/VERSION\1'"$1"'/' CMakeLists.txt
    sed -i 's/VERSION\(.*\)\w\.\w\.\w/VERSION\1'"$1"'/' meson.build
else
    usage
fi
