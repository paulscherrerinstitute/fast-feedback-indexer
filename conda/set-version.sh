function usage() {
    echo "usage: $0 [version]"
    echo
    echo "Replace version strings with new version"
    exit -1
}

if (($# == 0)); then
    grep "version =" meta.yaml
elif (($# == 1)); then
    echo "replace version strings by $1"
    sed -i 's/version = \"\w\+\.\w\+\.\w\+\"/version = \"'"$1"'\"/' meta.yaml
else
    usage
fi
