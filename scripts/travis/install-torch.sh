#/bin/bash
# Copyright (c) 2015-2016, NVIDIA CORPORATION.  All rights reserved.

set -e
set -x

if [ "$#" -ne 1 ];
then
    echo "Usage: $0 INSTALL_DIR"
    exit 1
fi
INSTALL_DIR=$1

if [ -d "$INSTALL_DIR" ] && [ -e "$INSTALL_DIR/install/bin/th" ]; then
    echo "Using cached build at $INSTALL_DIR ..."
    exit 0
fi

rm -rf $INSTALL_DIR
mkdir -p $INSTALL_DIR

# install Torch7
# instructions from: http://torch.ch/docs/getting-started.html
git clone https://github.com/torch/distro.git $INSTALL_DIR --recursive
cd $INSTALL_DIR
./install-deps
./install.sh -b

# install custom packages
${INSTALL_DIR}/install/bin/luarocks install tds
${INSTALL_DIR}/install/bin/luarocks install "https://raw.github.com/deepmind/torch-hdf5/master/hdf5-0-0.rockspec"
${INSTALL_DIR}/install/bin/luarocks install "https://raw.github.com/Sravan2j/lua-pb/master/lua-pb-scm-0.rockspec"
${INSTALL_DIR}/install/bin/luarocks install lightningmdb 0.9.18.1-1 LMDB_INCDIR=/usr/include LMDB_LIBDIR=/usr/lib/x86_64-linux-gnu
