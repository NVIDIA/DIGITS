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
NUM_THREADS=${NUM_THREADS:-4}
CAFFE_FORK=${CAFFE_FORK:-"NVIDIA"}
if [ ! -z "$CAFFE_BRANCH" ]; then
    CAFFE_BRANCH="--branch ${CAFFE_BRANCH}"
fi

if [ -d "$INSTALL_DIR" ] && [ -e "$INSTALL_DIR/build/tools/caffe" ]; then
    echo "Using cached build at $INSTALL_DIR ..."
    exit 0
fi

rm -rf $INSTALL_DIR

# get source
git clone https://github.com/${CAFFE_FORK}/caffe.git ${INSTALL_DIR} ${CAFFE_BRANCH} --depth 1

# configure project
pushd .
mkdir -p ${INSTALL_DIR}/build
cd ${INSTALL_DIR}/build
cmake .. -DCPU_ONLY=On -DBLAS=Open

# build
make --jobs=$NUM_THREADS

# mark cache
popd
WEEK=`date +%Y-%W`
echo $WEEK > ${INSTALL_DIR}/cache-version.txt

