# Install Caffe

To use Caffe in DIGITS, you must use [NVIDIA's fork](https://github.com/NVIDIA/caffe), version [`0.11`](https://github.com/NVIDIA/caffe/tree/caffe-0.11) or greater.

To enable multi-GPU training, install version [`0.12`](https://github.com/NVIDIA/caffe/tree/caffe-0.12).

To take advantage of the performance gains in cuDNN v3, install version [`0.13`](https://github.com/NVIDIA/caffe/tree/caffe-0.13).

## Grab the source

    % cd $HOME
    % git clone --branch caffe-0.13 https://github.com/NVIDIA/caffe.git

Set an environment variable so DIGITS knows where Caffe is installed (optional):

    % export CAFFE_HOME=${HOME}/caffe

## Install dependencies

If you are not on Ubuntu 14.04, you can try [Caffe's installation instructions](http://caffe.berkeleyvision.org/installation.html).
If you are, simply install these aptitude packages:

    % sudo apt-get install libprotobuf-dev libleveldb-dev libsnappy-dev libopencv-dev libhdf5-serial-dev libgflags-dev libgoogle-glog-dev liblmdb-dev protobuf-compiler libatlas-base-dev
    % sudo apt-get install --no-install-recommends libboost-all-dev
    % sudo apt-get install python-dev python-pip gfortran

### Python dependencies

These packages need to be installed:

    % cd $CAFFE_HOME
    % cat python/requirements.txt

The recommended installation method is using a **virtual environment** ([installation instructions](VirtualEnvironment.md)).

You may be tempted to install the requirements with `pip install -r python/requirements.txt`. Use the loop statement below to install the packages in order and avoid errors.

    % for req in $(cat python/requirements.txt); do pip install $req; done

If you want to install these packages *without* using a virtual environment, replace "pip install" with "**sudo** pip install".

## Build Caffe

    % cd $CAFFE_HOME
    % cp Makefile.config.example Makefile.config
    % make all --jobs=4
    % make py

NOTE: You may need to make some changes to `Makefile.config` to get Caffe to compile if you haven't installed CUDA or cuDNN.

(CMake instructions coming soon)
