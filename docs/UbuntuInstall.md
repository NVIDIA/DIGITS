# Ubuntu Installation

Deb packages are provided for easy installation on Ubuntu 14.04.

Packages are provided for major releases, but not minor ones (i.e. v3.0 and v4.0, but not v3.1).
If you want a newer version, you'll need to build at least DIGITS (and possibly also Caffe or Torch) from source ([instructions here](BuildDigits.md)).

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

Get access to machine learning packages from NVIDIA by downloading and installing the `cuda-repo-ubuntu1404` package (instructions above).
You could also get this package by downloading the `deb (network)` installer from the [CUDA downloads website](https://developer.nvidia.com/cuda-downloads).
This provides access to the repository containing packages like `cuda-toolkit-7-5` and `cuda-toolkit-7-0`, etc.

#### Machine Learning repository

Get access to machine learning packages from NVIDIA by downloading and installing the `nvidia-machine-learning-repo` package (instructions above).
This provides access to the repository containing packages like `digits`, `caffe-nv`, `torch`, `libcudnn4`, etc.

## DIGITS

Now that you have access to all the packages you need, installation is simple.
```sh
apt-get update
apt-get install digits
```
Through package dependency chains, this installs `digits`, `caffe-nv`, `libcudnn4` and many more packages for you automatically.

# Getting started

The DIGITS server should now be running at `http://localhost/`.
See comments below if you run into any issues.

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

#### Other

If you run into an issue not addressed here, try searching through the [GitHub issues](https://github.com/NVIDIA/DIGITS/issues) and/or the [user group](https://groups.google.com/d/forum/digits-users).
