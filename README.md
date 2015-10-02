# DIGITS

[![Build Status](https://travis-ci.org/NVIDIA/DIGITS.svg?branch=master)](https://travis-ci.org/NVIDIA/DIGITS)
[![Coverage Status](https://coveralls.io/repos/NVIDIA/DIGITS/badge.svg?branch=master)](https://coveralls.io/r/NVIDIA/DIGITS?branch=master)

DIGITS (the **D**eep Learning **G**PU **T**raining **S**ystem) is is a webapp for training deep learning models.

Table of Contents
=================
* [Installation](#installation)
  * [Prerequisites](#prerequisites)
  * [Install DIGITS](#install-digits)
* [Starting the server](#starting-the-server)
* [Get help](#get-help)

# Installation

DIGITS is only officially supported on Ubuntu 14.04. However, DIGITS has been successfully used on other Linux variants as well as on OSX.

## Prerequisites
DIGITS depends on several third-party libraries.

### CUDA

The CUDA toolkit (>= CUDA 6.5) is highly recommended, though not technically required.
* Download from the [CUDA website](https://developer.nvidia.com/cuda-downloads)
* Follow the installation instructions
* Don't forget to set your path. For example:
  * `CUDA_HOME=/usr/local/cuda`
  * `LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH`

For greater performance, you can also install cuDNN.
* Download from the [cuDNN website](https://developer.nvidia.com/cuDNN)
* Follow the installation instructions
* You can copy the files directly into your CUDA installation
    * `cp -a libcudnn* $CUDA_HOME/lib64/`
    * `cp cudnn.h $CUDA_HOME/include/`

### Deep learning frameworks

At least one deep learning framework backend is required.

* [Mandatory] Caffe (NVIDIA's fork) - [installation instructions](docs/InstallCaffe.md)
* [Optional] Torch7 - [installation instructions](docs/InstallTorch.md)

## Install DIGITS

### Grab the source

    % cd $HOME
    % git clone https://github.com/NVIDIA/DIGITS.git digits

Throughout the docs, we'll refer to your install location as `DIGITS_HOME` (`$HOME/digits` in this case), though you don't need to actually set that environment variable.

### Python dependencies

Several PyPI packages need to be installed. The recommended installation method is using a **virtual environment** ([installation instructions](docs/VirtualEnvironment.md)).

    % cd $DIGITS_HOME
    % pip install -r requirements.txt

If you want to install these packages *without* using a virtual environment, replace "pip install" with "**sudo** pip install".

### Other dependencies

DIGITS uses graphviz to visualize network architectures. You can safely skip this step if you don't want the feature.

    % sudo apt-get install graphviz

# Starting the server

You can run DIGITS in two ways:

### Development mode

    % ./digits-devserver

Starts a development server that listens on port 5000 (but you can change the port if you like - try running it with the --help flag).

Then, you can view your server at `http://localhost:5000/`.

### Production mode

    % ./digits-server

Starts a production server (gunicorn) that listens on port 8080 (`http://localhost:8080`). If you get any errors about an invalid configuration, use the development server first to set your configuration.

If you have installed the nginx.site to your nginx sites-enabled/ directory, then you can view your app at `http://localhost/`.

# Get help

### Installation issues
* First, check out the instructions below
* Then, ask questions on our [user group](https://groups.google.com/d/forum/digits-users)

### Usage questions
* First, check out the [Getting Started](docs/GettingStarted.md) page
* Then, ask questions on our [user group](https://groups.google.com/d/forum/digits-users)

### Bugs and feature requests
* Please let us know by [filing a new issue](https://github.com/NVIDIA/DIGITS/issues/new)
* Bonus points if you want to contribute by opening a [pull request](https://help.github.com/articles/using-pull-requests/)!
  * You will need to send a signed copy of the [Contributor License Agreement](CLA) to digits@nvidia.com before your change can be accepted.

