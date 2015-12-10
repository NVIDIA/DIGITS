# Building Torch

With v3.0, DIGITS now supports Torch7 as an optional alternative backend to Caffe.

> NOTE: Torch support is still experimental!

If you don't need a new version or custom build of Torch, you can still use deb packages to install the latest release.
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
* [Getting Started With Torch7 in DIGITS](#getting-started-with-torch7-in-digits)

## Prerequisites

### CUDA toolkit

To install the CUDA toolkit, first get access to the required repositories by following [these instructions](UbuntuInstall.md#repository-access).
Then install the toolkit with this command:
```sh
% sudo apt-get install cuda-toolkit-7-5
```
Any CUDA toolkit >= 6.5 should work.

### cuDNN

You can also install cuDNN via deb packages:
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

Install extra Lua packages:
```sh
% luarocks install image
% luarocks install "https://raw.github.com/deepmind/torch-hdf5/master/hdf5-0-0.rockspec"
```

## LMDB support

> NOTE: If you skip this section you will not be able to train Torch7 models on LMDB datasets!

Follow these instructions if you wish to use Torch7 to train networks using LMDB-encoded datasets in DIGITS. You may skip this section if you wish to only use HDF5-encoded datasets:
[LMDB installation instructions](BuildTorchLMDB.md)

## Getting Started With Torch7 in DIGITS

Follow [these instructions](GettingStartedTorch.md) for information on getting started with Torch7 in DIGITS.
