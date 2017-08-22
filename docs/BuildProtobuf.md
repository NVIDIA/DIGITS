# Building Protobuf

To be able to run Caffe and Tensorflow inside DIGITS side by side, protobuf 3 must be built from source. Caffe can be installed from debian packages but if it were to be run with tensorflow, some features such as python layers will not work properly. It is highly suggested to build protobuf from source to ensure DIGITS to work properly.

This guide is based on [google's protobuf installation guide](https://github.com/google/protobuf/blob/master/src/README.md).

## Dependencies

These Deb packages must be installed to build Protobuf 3
```
sudo apt-get install autoconf automake libtool curl make g++ git python-dev python-setuptools unzip
```

## Download Source

DIGITS is currently compatiable with Protobuf 3.2.x

```sh
# example location - can be customized
export PROTOBUF_ROOT=~/protobuf
git clone https://github.com/google/protobuf.git $PROTOBUF_ROOT -b '3.2.x'
```

## Building Protobuf

```sh
cd $PROTOBUF_ROOT
./autogen.sh
./configure
make "-j$(nproc)"
make install
ldconfig
cd python
python setup.py install --cpp_implementation
```

This will ensure that Protobuf 3 is installed.