# DiGiTS (Deep GPU Training System)

DiGiTS is is a webapp for training deep learning models.

## Installation

We install and run DiGiTS on Ubuntu 14.04.  We have successfully run DiGiTS on other Linux variants as well as OSX but at this time, only Ubuntu 14.04 is supported.

### Prerequisites
DiGiTS has several dependencies.

* CUDA
* cuDNN library
* Caffe – NVIDIA branch (version 0.10.0 or higher)
* Python packages
* Graphviz

1. CUDA (Either 6.5 or 7.0)

  * Download from the [CUDA website](https://developer.nvidia.com/cuda-downloads) and follow the installation instructions.

2. cuDNN (v2 Release Candidate 3 required)

  * Download from the [cuDNN website](https://developer.nvidia.com/cuDNN) and follow the installation instructions.

3. NVIDIA branch of Caffe (NVIDIA version 0.10.0)

Full installation directions are at [Caffe](http://caffe.berkeleyvision.org/installation.html). Condensed version is as follows:

Install caffe:
<pre>
% sudo apt-get install git
% cd $HOME
% git clone --branch v0.10.0 https://github.com/NVIDIA/caffe.git
% cd caffe
% sudo apt-get install libatlas-base-dev libatlas-dev libboost-all-dev libopencv-dev
% sudo apt-get install libprotobuf-dev libgoogle-glog-dev libgflags-dev protobuf-compiler
% sudo apt-get install libhdf5-dev libleveldb-dev liblmdb-dev libsnappy-dev
% sudo apt-get install python-pip gfortran
% cd python
% for req in $(cat requirements.txt); do sudo pip install $req; done
</pre>

Build caffe:
<pre>
% cd $HOME/caffe
% cp Makefile.config.example Makefile.config
% make all
% make py
% make test
% make runtest
</pre>

Set environment variables:
<pre>
% export CAFFE_HOME=${HOME}/caffe
</pre>

### Install DiGiTS

<pre>
% cd $HOME
% git clone https://github.com/NVIDIA/DIGITS.git digits
% cd digits
% sudo apt-get install graphviz gunicorn
% for req in $(cat requirements.txt); do sudo pip install $req; done
</pre>

## Starting the server

You can run DiGiTS in two ways:

1.  digits-devserver
        Starts a development server that listens on port 5000 (but you can
        change the port if you like - try running it with the --help flag).

2.  digits-server
        Starts a gunicorn app that listens on port 8080. If you have installed
        the nginx.site to your nginx sites-enabled/ directory, then you can
        view your app at http://localhost/.


Then, check out the [Getting Started](docs/GettingStarted.md) page.

