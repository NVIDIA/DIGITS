# Ubuntu Installation

Deb packages are provided for easy installation on Ubuntu 14.04.

## Prerequisites

NVIDIA driver version 346 or later.  If you need a driver go to http://www.nvidia.com/Download/index.aspx

## Repository access

Run the following commands to get access to the required repositories:
```sh
CUDA_REPO_PKG=cuda-repo-ubuntu1404_7.5-18_amd64.deb &&
    wget http://developer.download.nvidia.com/compute/cuda/repos/ubuntu1404/x86_64/$CUDA_REPO_PKG &&
    sudo dpkg -i $CUDA_REPO_PKG

ML_REPO_PKG=nvidia-machine-learning-repo_4.0-2_amd64.deb &&
    wget http://developer.download.nvidia.com/compute/machine-learning/repos/ubuntu1404/x86_64/$ML_REPO_PKG &&
    sudo dpkg -i $ML_REPO_PKG
```

#### CUDA repository

Get access to CUDA packages by downloading and installing the `deb (network)` installer from the [CUDA downloads website](https://developer.nvidia.com/cuda-downloads).
This provides access to the repository containing packages like `cuda-toolkit-7-5` and `cuda-toolkit-7-0`, etc.

#### Machine Learning repository

Get access to machine learning packages from NVIDIA by downloading and installing the `deb` installer from the [DIGITS website](https://developer.nvidia.com/digits).
This provides access to the repository containing packages like `digits`, `caffe-nv`, `torch`, `libcudnn4`, etc.

## DIGITS

Now that you have access to all the packages you need, installation is simple.
Through package dependency chains, this installs `digits`, `caffe-nv`, `libcudnn4` and many more packages for you automatically.
```sh
apt-get update
apt-get install digits
```

# Getting started

The DIGITS server should now be running at `http://localhost/`. See comments below for installation problems.

Now that you're up and running, check out the [getting started guide](GettingStarted.md).

## Troubleshooting

#### Configuration

If you have another server running on port 80 already, you may need to reconfigure DIGITS to use a different port.
```sh
% sudo dpkg-reconfigure digits
```

To make other configuration changes, try this (you probably want to leave most options as "unset" or "default" by hitting `ENTER` repeatedly):
```sh
% cd /usr/share/digits
# set new config
% sudo python -m digits.config.edit -v
# restart server
% sudo stop nvidia-digits-server
% sudo start nvidia-digits-server
```

#### Driver installations

If you try to install a new driver while the DIGITS server is running, you'll get an error about CUDA being in use.
Shut down the server before installing a driver, and then restart it afterwards:
```sh
% sudo stop nvidia-digits-server
# (install driver)
% sudo start nvidia-digits-server
```

#### Permissions

The DIGITS server runs as `www-data`, so keep in mind that prebuilt LMDB datasets used for generic models need to be readable by the `www-data` user.
