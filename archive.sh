#!/bin/sh

# Create a tar.xz archive of the code in the parent directory.
# This script only works if executed from the top directory of the repository.
#
# Usage: sh archive.sh
# Result: ../fast-feedback-indexer.tgz

if test -r archive.sh; then
    git archive --format=tgz --output=../fast-feedback-indexer.tgz --prefix=fast-feedback-indexer/ HEAD
else
    echo "Failed: execute this command from the top directory of the repository!"
fi
