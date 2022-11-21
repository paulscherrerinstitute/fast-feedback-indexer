#!/bin/sh

# Create a tar.xz archive of the code in the parent directory.
# This script only works if executed from the top directory of the repository.
#
# Usage: sh archive.sh
# Result: ../fast-feedback-indexer.tar.xz

if test -r archive.sh; then
    tar cvJf ../fast-feedback-indexer.tar.xz --exclude=build --exclude=install *
else
    echo "Failed: execute this command from the top directory of the repository!"
fi
