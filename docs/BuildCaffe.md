# Building Caffe

DIGITS requires a build of Caffe.
We officially only support recent releases from [NVIDIA/caffe](https://github.com/NVIDIA/caffe) (NVcaffe), but any recent build of [BVLC/caffe](https://github.com/BVLC/caffe) will probably work too.

## Dependencies

For best performance, you'll want:

* One or more NVIDIA GPUs ([details](InstallCuda.md#gpu))
* An NVIDIA driver ([details and installation instructions](InstallCuda.md#driver))
* A CUDA toolkit ([details and installation instructions](InstallCuda.md#cuda-toolkit))
* cuDNN ([download page](https://developer.nvidia.com/cudnn))

Install some dependencies with Deb packages:
```sh
sudo apt-get install --no-install-recommends build-essential cmake git gfortran libatlas-base-dev libboost-filesystem-dev libboost-python-dev libboost-system-dev libboost-thread-dev libgflags-dev libgoogle-glog-dev libhdf5-serial-dev libleveldb-dev liblmdb-dev libopencv-dev libprotobuf-dev libsnappy-dev protobuf-compiler python-all-dev python-dev python-h5py python-matplotlib python-numpy python-opencv python-pil python-pip python-protobuf python-scipy python-skimage python-sklearn
```

## Download source

```sh
# example location - can be customized
export CAFFE_ROOT=~/caffe
git clone https://github.com/NVIDIA/caffe.git $CAFFE_ROOT
```

Setting the `CAFFE_ROOT` environment variable will help DIGITS automatically detect your Caffe installation, but this is optional.

## Python packages

Several PyPI packages need to be installed:
```sh
sudo pip install -r $CAFFE_ROOT/python/requirements.txt
```

If you hit some errors about missing imports, then use this command to install the packages in order ([see discussion here](https://github.com/BVLC/caffe/pull/1950#issuecomment-76026969)):
```sh
cat $CAFFE_ROOT/python/requirements.txt | xargs -n1 sudo pip install
```

## Build

We recommend using CMake to configure Caffe rather than the raw Makefile build for automatic dependency detection:
```sh
cd $CAFFE_ROOT
mkdir build
cd build
cmake ..
make --jobs=4
```
