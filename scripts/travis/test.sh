#!/bin/bash
# Copyright (c) 2016, NVIDIA CORPORATION.  All rights reserved.
set -e

LOCAL_DIR=$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )
ROOT_DIR=$( dirname "$(dirname "$LOCAL_DIR")")

if [ ! -z "$DEB_BUILD" ]; then
    echo "Skipping for deb build"
    exit 0
fi

set -x

export CAFFE_ROOT=~/caffe
if [ -z "$DIGITS_TEST_FRAMEWORK" ] || [ "$DIGITS_TEST_FRAMEWORK" = "torch" ]; then
    export TORCH_ROOT=~/torch
fi
# Disable OpenMP multi-threading
export OMP_NUM_THREADS=1

cd $ROOT_DIR
./digits-test -v

