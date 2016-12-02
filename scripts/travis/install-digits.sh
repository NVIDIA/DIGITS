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

pip install -e $ROOT_DIR
pip install -e $ROOT_DIR/plugins/data/imageGradients
pip install -e $ROOT_DIR/plugins/view/imageGradients

