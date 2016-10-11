# Ubuntu Installation

Deb packages for major releases (i.e. v3.0 and v4.0 but not v3.1) are provided for easy installation on Ubuntu 14.04.
If these packages don't meet your needs, then you can follow [these instructions](BuildDigits.md) to build DIGITS and its dependencies from source.

## Prerequisites

You need an NVIDIA driver ([details and instructions](InstallCuda.md#driver)).

Run the following commands to get access to some package repositories:
```sh
# Access to CUDA packages
CUDA_REPO_PKG=cuda-repo-ubuntu1404_7.5-18_amd64.deb
wget http://developer.download.nvidia.com/compute/cuda/repos/ubuntu1404/x86_64/${CUDA_REPO_PKG} -O /tmp/${CUDA_REPO_PKG}
sudo dpkg -i /tmp/${CUDA_REPO_PKG}
rm -f /tmp/${CUDA_REPO_PKG}

# Access to Machine Learning packages
ML_REPO_PKG=nvidia-machine-learning-repo-ubuntu1404_4.0-2_amd64.deb
wget http://developer.download.nvidia.com/compute/machine-learning/repos/ubuntu1404/x86_64/${ML_REPO_PKG} -O /tmp/${ML_REPO_PKG}
sudo dpkg -i /tmp/${ML_REPO_PKG}
rm -f /tmp/${ML_REPO_PKG}

# Download new list of packages
sudo apt-get update
```

## Installation

Now that you have access to all the packages you need, installation is simple.
```sh
sudo apt-get install digits
```
Through package dependency chains, this installs `digits`, `caffe-nv`, `torch7-nv`, `libcudnn` and many more packages for you automatically.

## Getting started

The DIGITS server should now be running at `http://localhost/`.
See comments below if you run into any issues.

Now that you're up and running, check out the [Getting Started Guide](GettingStarted.md).

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
In particular, the entire chain of directories from `/` to your data must be readable by `www-data`.

#### Other

If you run into an issue not addressed here, try searching through the [GitHub issues](https://github.com/NVIDIA/DIGITS/issues) and/or the [user group](https://groups.google.com/d/forum/digits-users).
