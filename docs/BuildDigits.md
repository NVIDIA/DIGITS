# Building DIGITS

The following instructions will walk you through building the latest version of DIGITS from source.
**These instructions are for installation on Ubuntu 14.04 and 16.04.**

Alternatively, see [this guide](BuildDigitsWindows.md) for setting up DIGITS and Caffe on Windows machines.

Other platforms are not officially supported, but users have successfully installed DIGITS on Ubuntu 12.04, CentOS, OSX, and possibly more.
Since DIGITS itself is a pure Python project, installation is usually pretty trivial regardless of the platform.
The difficulty comes from installing all the required dependencies for Caffe, Torch7 , Tensorflow, and configuring the builds.
Doing so is your own adventure.

## Prerequisites

You need an NVIDIA driver ([details and instructions](InstallCuda.md#driver)).

Run the following commands to get access to some package repositories:
```sh
# For Ubuntu 14.04
CUDA_REPO_PKG=http://developer.download.nvidia.com/compute/cuda/repos/ubuntu1404/x86_64/cuda-repo-ubuntu1404_8.0.61-1_amd64.deb
ML_REPO_PKG=http://developer.download.nvidia.com/compute/machine-learning/repos/ubuntu1404/x86_64/nvidia-machine-learning-repo-ubuntu1404_4.0-2_amd64.deb

# For Ubuntu 16.04
CUDA_REPO_PKG=http://developer.download.nvidia.com/compute/cuda/repos/ubuntu1604/x86_64/cuda-repo-ubuntu1604_8.0.61-1_amd64.deb
ML_REPO_PKG=http://developer.download.nvidia.com/compute/machine-learning/repos/ubuntu1604/x86_64/nvidia-machine-learning-repo-ubuntu1604_1.0.0-1_amd64.deb

# Install repo packages
wget "$CUDA_REPO_PKG" -O /tmp/cuda-repo.deb && sudo dpkg -i /tmp/cuda-repo.deb && rm -f /tmp/cuda-repo.deb
wget "$ML_REPO_PKG" -O /tmp/ml-repo.deb && sudo dpkg -i /tmp/ml-repo.deb && rm -f /tmp/ml-repo.deb

# Download new list of packages
sudo apt-get update
```

## Dependencies

Install some dependencies with Deb packages:
```sh
sudo apt-get install --no-install-recommends git graphviz python-dev python-flask python-flaskext.wtf python-gevent python-h5py python-numpy python-pil python-pip python-scipy python-tk
```

Follow [these instructions](BuildCaffe.md) to build Caffe (**required**).

Follow [these instructions](BuildTorch.md) to build Torch7 (*suggested*).

Follow [these instructions](BuildTensorflow.md) to build Tensorflow (*suggseted*).

## Download source

```sh
# example location - can be customized
DIGITS_ROOT=~/digits
git clone https://github.com/NVIDIA/DIGITS.git $DIGITS_ROOT
```

Throughout the docs, we'll refer to your install location as `DIGITS_ROOT` (`~/digits` in this case), though you don't need to actually set that environment variable.

## Python packages

Several PyPI packages need to be installed:
```sh
sudo pip install -r $DIGITS_ROOT/requirements.txt
```

# [Optional] Enable support for plug-ins

DIGITS needs to be installed to enable loading data and visualization plug-ins:
```
sudo pip install -e $DIGITS_ROOT
```

# Starting the server

```sh
./digits-devserver
```

Starts a server at `http://localhost:5000/`.
```
$ ./digits-devserver --help
usage: __main__.py [-h] [-p PORT] [-d] [--version]

DIGITS development server

optional arguments:
  -h, --help            show this help message and exit
  -p PORT, --port PORT  Port to run app on (default 5000)
  -d, --debug           Run the application in debug mode (reloads when the
                        source changes and gives more detailed error messages)
  --version             Print the version number and exit
```

# Getting started

Now that you're up and running, check out the [Getting Started Guide](GettingStarted.md).

# Development

If you are interested in developing for DIGITS or work with its source code, check out the [Development Setup Guide](DevelopmentSetup.md)

## Troubleshooting

Most configuration options should have appropriate defaults.
Read [this doc](Configuration.md) for information about how to set a custom configuration for your server.
