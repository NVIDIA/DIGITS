#!/bin/bash
# Copyright (c) 2016, NVIDIA CORPORATION.  All rights reserved.
set -e

LOCAL_DIR=$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )
ROOT_DIR=$( dirname "$(dirname "$LOCAL_DIR")")

set -x

pip install -e $ROOT_DIR
pip install -e $ROOT_DIR/plugins/data/imageGradients
pip install -e $ROOT_DIR/plugins/view/imageGradients

