# DIGITS (Deep GPU Training System)

DIGITS is is a webapp for training deep learning models.

# Get help

#### Installation issues
* First, check out the instructions below
* Then, ask questions on our [user group](https://groups.google.com/d/forum/digits-users)

#### Usage questions
* First, check out the [Getting Started](docs/GettingStarted.md) page
* Then, ask questions on our [user group](https://groups.google.com/d/forum/digits-users)

#### Bugs and feature requests
* Please let us know by [filing a new issue](https://github.com/NVIDIA/DIGITS/issues/new)
* Bonus points if you want to contribute by opening a [pull request](https://help.github.com/articles/using-pull-requests/)!

# Installation

DIGITS is only officially supported on Ubuntu 14.04. However, DIGITS has been successfully used on other Linux variants as well as on OSX.

## Prerequisites
DIGITS has several dependencies.

* CUDA
* cuDNN library
* Caffe – NVIDIA branch (version 0.11.0)
* Python packages
* Graphviz

### CUDA (>= 6.5)

Download from the [CUDA website](https://developer.nvidia.com/cuda-downloads) and follow the installation instructions.

### cuDNN (>= v2)

Download from the [cuDNN website](https://developer.nvidia.com/cuDNN) and follow the installation instructions.

### NVIDIA's fork of Caffe (NVIDIA version 0.11.0)

Detailed installation instructions are available on [caffe's installation page](http://caffe.berkeleyvision.org/installation.html). Condensed version is as follows:

#### Installing caffe prerequisites on Linux

    % sudo apt-get install git
    % cd $HOME
    % git clone --branch v0.11.0 https://github.com/NVIDIA/caffe.git
    % cd caffe
    % sudo apt-get install libprotobuf-dev libleveldb-dev libsnappy-dev libopencv-dev libboost-all-dev libhdf5-serial-dev libgflags-dev libgoogle-glog-dev liblmdb-dev protobuf-compiler libatlas-base-dev
    % sudo apt-get install python-dev python-pip gfortran
    % cd python
    % for req in $(cat requirements.txt); do sudo pip install $req; done

#### Installing caffe prerequisites on Mac OS

If you have [homebrew](http://brew.sh/) installed, you can follow the instructions from [caffe's page](http://caffe.berkeleyvision.org/install_osx.html)

Before you build caffe you may need to specify the include directory for cudnn in Makefile.config. Also remember to select the correct option for BLAS (atlas, mkl or open). We installed openblas using the commands above.

#### Build caffe:

    % cd $HOME/caffe
    % cp Makefile.config.example Makefile.config
    % make all --jobs=4
    % make py

Set environment variables:

    % export CAFFE_HOME=${HOME}/caffe

## Install DIGITS

    % cd $HOME
    % git clone https://github.com/NVIDIA/DIGITS.git digits
    % cd digits
    % sudo apt-get install graphviz gunicorn
    % for req in $(cat requirements.txt); do sudo pip install $req; done

# Starting the server

You can run DIGITS in two ways:

### digits-devserver

Starts a development server that listens on port 5000 (but you can
change the port if you like - try running it with the --help flag).

### digits-server

Starts a gunicorn app that listens on port 8080. If you have installed
the nginx.site to your nginx sites-enabled/ directory, then you can
view your app at http://localhost/.

# Usage

Check out the [Getting Started](docs/GettingStarted.md) page.
