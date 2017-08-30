#!/bin/bash
# Copyright (c) 2016-2017, NVIDIA CORPORATION.  All rights reserved.
set -e

LOCAL_DIR=$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )

if [ "$#" -ne 1 ];
then
    echo "Usage: $0 INSTALL_DIR"
    exit 1
fi

INSTALL_DIR=$(readlink -f "$1")
if [ -d "$INSTALL_DIR" ] && [ -f "$INSTALL_DIR/src/protoc" ]; then
    echo "Using cached build at $INSTALL_DIR ..."
else
    rm -rf "$INSTALL_DIR"
    git clone https://github.com/google/protobuf.git "$INSTALL_DIR" -b '3.2.x' --depth 1
    cd "$INSTALL_DIR"

    ./autogen.sh
    ./configure
    make "-j$(nproc)"
fi

cd "$INSTALL_DIR"
sudo make install
sudo ldconfig

cd python
python setup.py install --cpp_implementation
