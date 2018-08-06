#!/bin/bash
# Copyright (c) 2015-2017, NVIDIA CORPORATION.  All rights reserved.
set -e

LOCAL_DIR=$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )

if [ "$#" -ne 1 ];
then
    echo "Usage: $0 INSTALL_DIR"
    exit 1
fi

INSTALL_DIR=$(readlink -f "$1")
"$LOCAL_DIR/bust-cache.sh" "$INSTALL_DIR"
if [ -d "$INSTALL_DIR" ] && [ -e "$INSTALL_DIR/build/tools/caffe" ]; then
    echo "Using cached build at $INSTALL_DIR ..."
    exit 0
fi
rm -rf "$INSTALL_DIR"

CAFFE_FORK=${CAFFE_FORK:-"NVIDIA"}
if [ ! -z "$CAFFE_BRANCH" ]; then
    CAFFE_BRANCH="--branch ${CAFFE_BRANCH}"
fi

set -x

if [ "$CAFFE_FORK" == "NVIDIA" ]; then
    # get source
    git clone "https://github.com/${CAFFE_FORK}/caffe.git" "$INSTALL_DIR" $CAFFE_BRANCH --depth 1
    # configure project
    cd "$INSTALL_DIR"
    git fetch --all --tags --prune
    git checkout "tags/$NV_CAFFE_TAG"
else
    # get source
    git clone "https://github.com/${CAFFE_FORK}/caffe.git" "$INSTALL_DIR" 
    # configure project
    cd "$INSTALL_DIR"
    git checkout "$BVLC_CAFFE_COMMIT"
fi
mkdir -p build
cd build
cmake .. -DCPU_ONLY=1 -DBLAS=Open

# build
make --jobs="$(nproc)"

# mark cache
WEEK=$(date +%Y-%W)
echo "$WEEK" > "${INSTALL_DIR}/cache-version.txt"

