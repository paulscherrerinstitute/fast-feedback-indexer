function usage() {
    echo "usage: $0 [component [version] | components] "
    echo
    echo "Show or replace version strings with new version"
    exit -1
}

COMPONENTS=($(find . -mindepth 2 -name set-version.sh))

function list_components() {
    for SCRIPT in "${COMPONENTS[@]}"; do
        echo "$SCRIPT"
    done
}

function unknown_component() {
    echo "unknown component - use one of"
    list_components
    exit -1
}

if (($# == 0)); then
    for SCRIPT in "${COMPONENTS[@]}"; do
        echo "COMPONENT: $SCRIPT"
        pushd $(dirname "$SCRIPT") &> /dev/null
        ./"$(basename "$SCRIPT")"
        popd &> /dev/null
    done
elif (($# > 2)); then
    usage
elif [ "$1" == "components" ]; then
    list_components
else
    SCRIPT=""
    for COMPONENT in "${COMPONENTS[@]}"; do
        if [ "$1" == "$COMPONENT" ]; then
            SCRIPT="$1"
            break
        fi
    done
    if [ -z "$SCRIPT" ]; then
        unknown_component
    fi
    pushd $(dirname "$SCRIPT") &> /dev/null
    eval "./$(basename "$SCRIPT") $2"
    popd &> /dev/null
fi
