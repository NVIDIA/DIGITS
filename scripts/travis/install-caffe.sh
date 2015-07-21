#/usr/bin/env bash
# Copyright (c) 2015, NVIDIA CORPORATION.  All rights reserved.

set -e

if [ "$#" -ne 1 ];
then
    echo "Usage: $0 INSTALL_DIR"
    exit 1
fi
INSTALL_DIR=$1
mkdir -p $INSTALL_DIR

CAFFE_TAG="caffe-0.11"
CAFFE_URL="https://github.com/NVIDIA/caffe.git"

# Get source
git clone --depth 1 --branch $CAFFE_TAG $CAFFE_URL $INSTALL_DIR
cd $INSTALL_DIR

# Install dependencies
sudo -E ./scripts/travis/travis_install.sh
# change permissions for installed python packages
sudo chown travis:travis -R /home/travis/miniconda

# Build source
cp Makefile.config.example Makefile.config
sed -i 's/# CPU_ONLY/CPU_ONLY/g' Makefile.config
sed -i 's/USE_CUDNN/#USE_CUDNN/g' Makefile.config
make --jobs=$NUM_THREADS all
make --jobs=$NUM_THREADS pycaffe

# Install python dependencies
# conda (fast)
conda install --yes cython nose ipython h5py pandas python-gflags
# pip (slow)
for req in $(cat python/requirements.txt); do pip install $req; done

