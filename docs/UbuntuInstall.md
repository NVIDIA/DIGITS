# Ubuntu Installation

Debian packages are provided for easy installation on Ubuntu 14.04.

## Prerequisites

NVIDIA driver version 346 or later.  If you need a driver go to http://www.nvidia.com/Download/index.aspx

## Repository access

Run the following commands to get access to the required repositories:
```sh
CUDA_REPO_PKG=cuda-repo-ubuntu1404_7.0-28_amd64.deb &&
    wget http://developer.download.nvidia.com/compute/cuda/repos/ubuntu1404/x86_64/$CUDA_REPO_PKG &&
    sudo dpkg -i $CUDA_REPO_PKG

ML_REPO_PKG=nvidia-machine-learning-repo_4.0-1_amd64.deb &&
    wget http://developer.download.nvidia.com/compute/machine-learning/repos/ubuntu1404/x86_64/$ML_REPO_PKG &&
    sudo dpkg -i $ML_REPO_PKG
```

#### CUDA repository

Get access to CUDA packages by downloading and installing the `deb (network)` installer from the [CUDA downloads website](https://developer.nvidia.com/cuda-downloads).
This provides access to the repository containing packages like `cuda-toolkit-7-0` and `cuda-cudart-7-0`, etc.

#### Machine Learning repository

Get access to machine learning packages from NVIDIA by downloading and installing the `deb` installer from the [DIGITS website](https://developer.nvidia.com/digits).
This provides access to the repository containing packages like `digits`, `caffe-nv`, `torch`, `libcudnn4`, etc.

## DIGITS

Now that you have access to all the packages you need, installation is simple.

```sh
apt-get update
apt-get install digits
```
Through the dependency chains, this installs `digits`, `caffe-nv`, `libcudnn4`, `cuda-cudart-7-0`, and many more packages for you automatically.

# Getting started

Now that you're up and running, check out the [getting started guide](GettingStarted.md).
