function usage() {
    echo "usage: $0 component [version]"
    echo
    echo "Show or replace version strings with new version"
    exit -1
}

COMPONENTS=($(find . -mindepth 2 -name set-version.sh))

function unknown_component() {
    echo "unknown component - use one of"
    for SCRIPT in "${COMPONENTS[@]}"; do
        echo "$SCRIPT"
    done
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
else
    SCRIPT="unknown_component"
    for COMPONENT in "${COMPONENTS[@]}"; do
        if [ "$1" == "$COMPONENT" ]; then
            SCRIPT="$1"
            break
        fi
    done
    pushd $(dirname "$SCRIPT") &> /dev/null
    eval "./$(basename "$SCRIPT") $2"
    popd &> /dev/null
fi
