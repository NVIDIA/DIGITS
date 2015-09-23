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

CAFFE_BRANCH="caffe-0.13"
CAFFE_URL="https://github.com/NVIDIA/caffe.git"

# Get source
git clone --depth 1 --branch $CAFFE_BRANCH $CAFFE_URL $INSTALL_DIR
cd $INSTALL_DIR

# Install dependencies
sudo -E ./scripts/travis/travis_install.sh
# change permissions for installed python packages
sudo chown travis:travis -R /home/travis/miniconda
sudo chown travis:travis -R /home/travis/.cache

# Build source
cp Makefile.config.example Makefile.config
sed -i 's/# CPU_ONLY/CPU_ONLY/g' Makefile.config
sed -i 's/USE_CUDNN/#USE_CUDNN/g' Makefile.config
sed -i 's/# WITH_PYTHON_LAYER/WITH_PYTHON_LAYER/g' Makefile.config

# Use miniconda
sed -i 's/# ANACONDA_HOME/ANACONDA_HOME/' Makefile.config
sed -i 's/# PYTHON_INCLUDE/PYTHON_INCLUDE/' Makefile.config
sed -i 's/# $(ANACONDA_HOME)/$(ANACONDA_HOME)/' Makefile.config
sed -i 's/# PYTHON_LIB/PYTHON_LIB/' Makefile.config
sed -i 's/ANACONDA/MINICONDA/g' Makefile.config
sed -i 's/Anaconda/Miniconda/g' Makefile.config
sed -i 's/anaconda/miniconda/g' Makefile.config
echo 'LINKFLAGS += -Wl,-rpath,/home/travis/miniconda/lib' >> Makefile.config

# compile
make --jobs=$NUM_THREADS all
make --jobs=$NUM_THREADS pycaffe

# Install python dependencies
# conda (fast)
conda install --yes cython nose ipython h5py pandas python-gflags
# pip (slow)
for req in $(cat python/requirements.txt); do
    pip install $req
done

