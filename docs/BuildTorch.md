# Building Torch

With v3.0, DIGITS now supports Torch7 as an optional alternative backend to Caffe.

> NOTE: Torch support is still experimental!

If you don't need a new version or custom build of Torch, you can still use Deb packages to install the latest release.
Follow [these instructions](UbuntuInstall.md#repository-access) to gain access to the required repositories, and then use this command to install:
```sh
% sudo apt-get install torch7-nv
```
Otherwise, follow these instructions to build from source.

Table of Contents
=================
* [Prerequisites](#prerequisites)
* [Torch installer](#torch-installer)
* [Luarocks dependencies](#luarocks-dependencies)
* [LMDB support](#lmdb-support)
* [NCCL support](#nccl-support)
* [Getting Started With Torch7 in DIGITS](#getting-started-with-torch7-in-digits)

## Prerequisites

### CUDA toolkit

To install the CUDA toolkit with Deb packages, first get access to the required repositories by following [these instructions](UbuntuInstall.md#repository-access).
Then, install your toolkit (any version >= 6.5 is fine):
```sh
# If you already have a driver installed
% sudo apt-get install cuda-toolkit-7-5

# If you need a driver
% sudo apt-get install cuda-7-5
```
For more information, see [InstallCuda.md](InstallCuda.md).

### cuDNN

You can also install cuDNN 4 with a Deb package:
```sh
% sudo apt-get install libcudnn4-dev
```

## Torch installer

Follow these instructions to install Torch7 on Mac OS X and Ubuntu 12+:

http://torch.ch/docs/getting-started.html

After installing Torch, you may consider refreshing your bash by doing `source ~/.bashrc`. This will conveniently add the Torch `th` executable to your executable search path, which will allow DIGITS to find it automatically.

## Luarocks dependencies

To use Torch7 in DIGITS, you need to install a few extra dependencies.

If you haven't done so already, install the HDF5 package:
```sh
% sudo apt-get install libhdf5-serial-dev
```

Install extra luarocks packages:
```sh
% luarocks install image
% luarocks install "https://raw.github.com/deepmind/torch-hdf5/master/hdf5-0-0.rockspec"
```

## LMDB support

> NOTE: If you skip this section you will not be able to train Torch7 models on LMDB datasets!

Follow these instructions if you wish to use Torch7 to train networks using LMDB-encoded datasets in DIGITS. You may skip this section if you wish to only use HDF5-encoded datasets:
[LMDB installation instructions](BuildTorchLMDB.md)

## NCCL support

[NCCL](https://github.com/NVIDIA/nccl) is a library of primitives for multi-GPU communication.
You may consider installing the [nccl.torch](https://github.com/ngimel/nccl.torch) module if you wish to speed up
multi-GPU training, although this module is not strictly required to enable multi-GPU training in Torch7.

Download and build NCCL:
```sh
% git clone https://github.com/NVIDIA/nccl.git
% cd nccl
% make CUDA_HOME=/usr/local/cuda test
```

> NOTE: if the above command fails due to missing libraries you may explicitly point the makefile to the location of your NVidia driver. For example:

```sh
% make CUDA_HOME=/usr/local/cuda LIBRARY_PATH=/usr/lib/nvidia-352 test
```

Add the path to the NCCL library to your library path:
```sh
% export LD_LIBRARY_PATH=.../nccl/build/lib:$LD_LIBRARY_PATH
```

Install the `nccl.torch` luarocks package:
```sh
% luarocks install "https://raw.githubusercontent.com/ngimel/nccl.torch/master/nccl-scm-1.rockspec"
```

Verify your installation of nccl.torch:
```sh
gheinrich@ubuntu:~/ws/nccl.torch$ th
  ______             __   |  Torch7
 /_  __/__  ________/ /   |  Scientific computing for Lua.
  / / / _ \/ __/ __/ _ \  |  Type ? for help
 /_/  \___/_/  \__/_//_/  |  https://github.com/torch
                          |  http://torch.ch
th> require('nccl')
{
  createCommunicators : function: 0x41084588
  reduceScatter : function: 0x41084888
  allGather : function: 0x41084860
  C : userdata: 0x410844a8
  bcast : function: 0x41084830
  communicators : {...}
  allReduce : function: 0x410844c0
  reduce : function: 0x41084718
}
```

## Getting Started With Torch7 in DIGITS

Follow [these instructions](GettingStartedTorch.md) for information on getting started with Torch7 in DIGITS.
