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
"${LOCAL_DIR}/bust-cache.sh" "$INSTALL_DIR"
if [ -d "$INSTALL_DIR" ] && [ -e "$INSTALL_DIR/install/bin/th" ]; then
    echo "Using cached build at $INSTALL_DIR ..."
    exit 0
fi
rm -rf "$INSTALL_DIR"

NUMTHREADS=$(nproc)
export MAKEFLAGS="-j$NUMTHREADS"

set -x

# get source
git clone https://github.com/torch/distro.git "$INSTALL_DIR" --recursive
cd "$INSTALL_DIR"

# Redirect build output to a log and only show it if an error occurs
# Otherwise there is too much output for TravisCI to display properly
LOG_FILE="$LOCAL_DIR/torch-install.log"
./install.sh -b >"$LOG_FILE" 2>&1 || (cat "$LOG_FILE" && false)

# install custom packages
"${INSTALL_DIR}/install/bin/luarocks" install tds
# you can't just install "hdf5" because it will install "lua-hdf5" instead of "torch-hdf5"
"${INSTALL_DIR}/install/bin/luarocks" install "https://raw.github.com/deepmind/torch-hdf5/master/hdf5-0-0.rockspec"
"${INSTALL_DIR}/install/bin/luarocks" install "https://raw.github.com/Neopallium/lua-pb/master/lua-pb-scm-0.rockspec"
"${INSTALL_DIR}/install/bin/luarocks" install lightningmdb 0.9.18.1-1 LMDB_INCDIR=/usr/include LMDB_LIBDIR=/usr/lib/x86_64-linux-gnu

# mark cache
WEEK=$(date +%Y-%W)
echo "$WEEK" >"${INSTALL_DIR}/cache-version.txt"

