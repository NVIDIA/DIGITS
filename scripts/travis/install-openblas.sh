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
if [ -d "$INSTALL_DIR" ] && ls "$INSTALL_DIR/"*.so >/dev/null 2>&1; then
    echo "Using cached build at $INSTALL_DIR ..."
else
    rm -rf "$INSTALL_DIR"
    git clone https://github.com/xianyi/OpenBLAS.git "$INSTALL_DIR" -b v0.2.18 --depth 1
    cd "$INSTALL_DIR"

    # Redirect build output to a log and only show it if an error occurs
    # Otherwise there is too much output for TravisCI to display properly
    LOG_FILE="$LOCAL_DIR/openblas-build.log"
    make NO_AFFINITY=1 USE_OPENMP=1 >"$LOG_FILE" 2>&1 || (cat "$LOG_FILE" && false)
fi

cd "$INSTALL_DIR"
sudo make install PREFIX=/usr/local
