#/usr/bin/env bash
# Copyright (c) 2015, NVIDIA CORPORATION.  All rights reserved.

set -e
set -x

if [ "$#" -ne 1 ];
then
    echo "Usage: $0 INSTALL_DIR"
    exit 1
fi
INSTALL_DIR=$1
mkdir -p $INSTALL_DIR

# install Torch7
# instructions from: http://torch.ch/docs/getting-started.html
curl -s https://raw.githubusercontent.com/torch/ezinstall/master/install-deps | bash
git clone https://github.com/torch/distro.git $INSTALL_DIR --recursive
cd $INSTALL_DIR; ./install.sh -b

# install custom packages
${INSTALL_DIR}/install/bin/luarocks install image
${INSTALL_DIR}/install/bin/luarocks install "https://raw.github.com/deepmind/torch-hdf5/master/hdf5-0-0.rockspec"
${INSTALL_DIR}/install/bin/luarocks install "https://raw.github.com/Sravan2j/lua-pb/master/lua-pb-scm-0.rockspec"
${INSTALL_DIR}/install/bin/luarocks install lightningmdb LMDB_INCDIR=/usr/local/include LMDB_LIBDIR=/usr/local/lib

