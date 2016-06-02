#!/bin/bash
# Copyright (c) 2015-2016, NVIDIA CORPORATION.  All rights reserved.

set -e
set -x

if [ "$#" -ne 1 ];
then
    echo "Usage: $0 INSTALL_DIR"
    exit 1
fi

INSTALL_DIR=$1
NUM_THREADS=${NUM_THREADS-4}
CAFFE_URL=https://github.com/NVIDIA/caffe.git
CAFFE_BRANCH=caffe-0.14

if [ -d "$INSTALL_DIR" ] && [ -e "$INSTALL_DIR/build/tools/caffe" ]; then
    echo "Using cached build at $INSTALL_DIR ..."
    exit 0
fi

rm -rf $INSTALL_DIR

# get source
git clone --branch ${CAFFE_BRANCH} --depth 1 ${CAFFE_URL} ${INSTALL_DIR}

# configure project
mkdir -p ${INSTALL_DIR}/build
cd ${INSTALL_DIR}/build
cmake .. -DCPU_ONLY=On -DBLAS=Open

# build
make --jobs=$NUM_THREADS
