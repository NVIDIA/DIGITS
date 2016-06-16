#!/bin/sh
# Copyright (c) 2016, NVIDIA CORPORATION.  All rights reserved.

set -e
set -x

sudo apt-get update
sudo apt-get install -y --no-install-recommends \
    build-essential \
    cmake \
    cython \
    git \
    gfortran \
    graphviz \
    libboost-filesystem-dev \
    libboost-python-dev \
    libboost-system-dev \
    libboost-thread-dev \
    libgflags-dev \
    libgoogle-glog-dev \
    libhdf5-serial-dev \
    libleveldb-dev \
    liblmdb-dev \
    libopenblas-dev \
    libopencv-dev \
    libprotobuf-dev \
    libsnappy-dev \
    protobuf-compiler \
    python-dev \
    python-flask \
    python-gevent \
    python-gflags \
    python-h5py \
    python-mock \
    python-nose \
    python-numpy \
    python-pil \
    python-pip \
    python-protobuf \
    python-psutil \
    python-requests \
    python-scipy \
    python-six \
    python-skimage

