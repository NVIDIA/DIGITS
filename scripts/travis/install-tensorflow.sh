#!/bin/bash
# Copyright (c) 2016, NVIDIA CORPORATION.  All rights reserved.
set -e

LOCAL_DIR=$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )

if [ ! -z "$DEB_BUILD" ]; then
    echo "Skipping for deb build"
    exit 0
fi

set -x

pip install --upgrade https://storage.googleapis.com/tensorflow/linux/cpu/tensorflow-0.11.0rc0-cp27-none-linux_x86_64.whl

