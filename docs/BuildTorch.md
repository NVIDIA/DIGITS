# Building Torch

DIGITS recommends a build of Torch7 to use as an alternative backend to Caffe, though it is not required.

## Dependencies

For best performance, you'll want:

* One or more NVIDIA GPUs ([details](InstallCuda.md#gpu))
* An NVIDIA driver ([details and installation instructions](InstallCuda.md#driver))
* A CUDA toolkit ([details and installation instructions](InstallCuda.md#cuda-toolkit))
* cuDNN ([download page](https://developer.nvidia.com/cudnn))

Install some dependencies with Deb packages:
```sh
sudo apt-get install --no-install-recommends git software-properties-common
```

## Basic install

These instructions are based on [the official Torch instructions](http://torch.ch/docs/getting-started.html).
```sh
# example location - can be customized
export TORCH_ROOT=~/torch

git clone https://github.com/torch/distro.git $TORCH_ROOT --recursive
cd $TORCH_ROOT
./install-deps
./install.sh -b
source ~/.bashrc
```

## Extra dependencies

DIGITS requires you to install a few extra dependencies.

```sh
sudo apt-get install --no-install-recommends libhdf5-serial-dev liblmdb-dev
luarocks install tds
luarocks install "https://raw.github.com/deepmind/torch-hdf5/master/hdf5-0-0.rockspec"
luarocks install "https://raw.github.com/Neopallium/lua-pb/master/lua-pb-scm-0.rockspec"
luarocks install lightningmdb 0.9.18.1-1 LMDB_INCDIR=/usr/include LMDB_LIBDIR=/usr/lib/x86_64-linux-gnu
# If you have installed NCCL
luarocks install "https://raw.githubusercontent.com/ngimel/nccl.torch/master/nccl-scm-1.rockspec"
```

## Getting Started With Torch7 in DIGITS

Follow [these instructions](GettingStartedTorch.md) for information on getting started with Torch7 in DIGITS.
