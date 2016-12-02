#!/bin/bash
# Copyright (c) 2016, NVIDIA CORPORATION.  All rights reserved.
set -e

LOCAL_DIR=$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )
ROOT_DIR=$( dirname "$(dirname "$LOCAL_DIR")")

if [ -z "$DEB_BUILD" ]; then
    echo "Skipping for non-deb build"
    exit 0
fi

set -x

git fetch --tags
$ROOT_DIR/packaging/deb/build.sh

