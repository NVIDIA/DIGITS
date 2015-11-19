# Install Caffe

To use Caffe in DIGITS, you must use [NVIDIA's fork](https://github.com/NVIDIA/caffe), version [`0.11`](https://github.com/NVIDIA/caffe/tree/caffe-0.11) or greater.

To enable multi-GPU training, install version [`0.12`](https://github.com/NVIDIA/caffe/tree/caffe-0.12).

For cuDNN v3, install version [`0.13`](https://github.com/NVIDIA/caffe/tree/caffe-0.12).

## Grab the source

    % cd $HOME
    % git clone --branch caffe-0.12 https://github.com/NVIDIA/caffe.git

Set an environment variable so DIGITS knows where Caffe is installed (optional):

    % export CAFFE_HOME=${HOME}/caffe

## Install dependencies

If you are not on Ubuntu 14.04, you can try [Caffe's installation instructions](http://caffe.berkeleyvision.org/installation.html).
If you are, simply install these aptitude packages:

    % sudo apt-get install \
        libgflags-dev libgoogle-glog-dev \
        libopencv-dev \
        libleveldb-dev libsnappy-dev liblmdb-dev libhdf5-serial-dev \
        libprotobuf-dev protobuf-compiler \
        libatlas-base-dev \
        python-dev python-pip python-numpy gfortran
    % sudo apt-get install --no-install-recommends libboost-all-dev

### Python dependencies

These packages need to be installed:

    % cd $CAFFE_HOME
    % cat python/requirements.txt

The recommended installation method is using a **virtual environment** ([installation instructions](VirtualEnvironment.md)).

You may be tempted to install the requirements with `pip install -r python/requirements.txt`. Use the loop statement below to install the packages in order and avoid errors.

    % for req in $(cat python/requirements.txt); do pip install $req; done

If you want to install these packages *without* using a virtual environment, replace "pip install" with "**sudo** pip install".

## Build Caffe

#### Using CMake

You can use [CMake](http://www.cmake.org/) to configure your build for you.
It will try to find the correct paths to every library needed to build Caffe.

    % cd $CAFFE_HOME
    % mkdir build
    % cd build
    % cmake ..
    % make --jobs=4

If you are using [cuDNN](https://developer.nvidia.com/cudnn) (for faster performance) and [CNMeM](https://github.com/NVIDIA/cnmem) (better GPU memory utilization for faster performance and fewer out-of-memory-errors), then you may want to configure CMake like this:

    % export CUDNN_HOME=/path/to/cudnn
    % export CNMEM_HOME=/path/to/cnmem
    % cmake \
      -DUSE_CNMEM=ON \
      -DCUDNN_INCLUDE=${CUDNN_HOME}/include -DCUDNN_LIBRARY=${CUDNN_HOME}/lib64/libcudnn.so \
      -DCNMEM_INCLUDE=${CNMEM_HOME}/include -DCNMEM_LIBRARY=${CNMEM_HOME}/build/libcnmem.so \
      ..

#### Using Make

You can also use Make directly.

    % cd $CAFFE_HOME
    % cp Makefile.config.example Makefile.config
    % make all --jobs=4
    % make py

NOTE: You may need to make some changes to `Makefile.config` to get Caffe to compile if you haven't installed CUDA or cuDNN.

