# Building Caffe

DIGITS requires a build of Caffe.
We officially only support recent releases from [NVIDIA/caffe](https://github.com/NVIDIA/caffe) (NVcaffe), but any recent build of [BVLC/caffe](https://github.com/NVIDIA/caffe) will probably work too.

## Dependencies

For best performance, you'll want:

* One or more NVIDIA GPUs ([details](InstallCuda.md#gpu))
* An NVIDIA driver ([details and installation instructions](InstallCuda.md#driver))
* A CUDA toolkit ([details and installation instructions](InstallCuda.md#cuda-toolkit))
* cuDNN ([download page](https://developer.nvidia.com/cudnn))

Install some dependencies with Deb packages:
```sh
sudo apt-get install --no-install-recommends build-essential cmake git gfortran libatlas-base-dev libboost-all-dev libgflags-dev libgoogle-glog-dev libhdf5-serial-dev libleveldb-dev liblmdb-dev libopencv-dev libprotobuf-dev libsnappy-dev protobuf-compiler python-all-dev python-dev python-h5py python-matplotlib python-numpy python-opencv python-pil python-pip python-protobuf python-scipy python-skimage python-sklearn
```

## Download source

```sh
# example location - can be customized
export CAFFE_HOME=~/caffe
git clone https://github.com/NVIDIA/caffe.git $CAFFE_HOME
```

Setting the `CAFFE_HOME` environment variable will help DIGITS automatically detect your Caffe installation, but this is optional.

## Python packages

Several PyPI packages need to be installed:
```sh
sudo pip install -r $CAFFE_HOME/python/requirements.txt
```

If you hit some errors about missing imports, then use this command to install the packages in order ([see discussion here](https://github.com/BVLC/caffe/pull/1950#issuecomment-76026969)):
```sh
cat $CAFFE_HOME/python/requirements.txt | xargs -n1 sudo pip install
```

### Check gcc version
On Ubuntu 16.04, you would need to make sure your gcc/g++ is of version 5 (5.4.0) not 4.9 or lower. Otherwise, you would encounter issues compiling Caffee, such as https://github.com/BVLC/caffe/issues/3438.
```
gcc --version
gcc (Ubuntu 5.4.0-6ubuntu1~16.04.2) 5.4.0 20160609
```
## Build

We recommend using CMake to configure Caffe rather than the raw Makefile build for automatic dependency detection:
```sh
cd $CAFFE_HOME
mkdir build
cd build
cmake ..
make --jobs=4
```
