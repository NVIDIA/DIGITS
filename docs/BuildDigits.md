# Building DIGITS

The preferred installation method for DIGITS is via deb packages ([instructions](UbuntuInstall.md)).
If you need to use a newer version of DIGITS or a custom build of NVcaffe, then you can use the instructions below to build from source.

Please note that Ubuntu **14.04 is the only officially supported OS** at this time, although DIGITS has been successfully used on other Linux variants as well as on OSX.
If you want to use DIGITS on an alternative OS, your main obstacle will be building Caffe.
Please refer to BVLC's [installation docs](http://caffe.berkeleyvision.org/installation.html), [user group](https://groups.google.com/d/forum/caffe-users) and/or [GitHub issues](https://github.com/BVLC/caffe/issues).

## Prerequisites

Unless you build Caffe and Torch without CUDA, you'll need an NVIDIA driver version 346 or later. You can get one from the [NVIDIA driver website](http://www.nvidia.com/Download/index.aspx).

You'll also need a few basic packages:
```sh
% sudo apt-get install python-dev python-pip graphviz
```

## Download source
```sh
% cd $HOME
% git clone https://github.com/NVIDIA/DIGITS.git digits
```

Throughout the docs, we'll refer to your install location as `DIGITS_HOME` (`$HOME/digits` in this case), though you don't need to actually set that environment variable.

## Python packages

Several PyPI packages need to be installed.
```sh
% cd $DIGITS_HOME
% sudo pip install -r requirements.txt
```

To speed up installation, you could install most of these via apt-get packages first.
```sh
% sudo apt-get install python-pil python-numpy python-scipy python-protobuf python-gevent python-Flask python-flaskext.wtf gunicorn python-h5py
```

## Caffe

DIGITS requires [NVIDIA's fork of Caffe](https://github.com/NVIDIA/caffe), which is sometimes referred to as either "NVcaffe" or "caffe-nv".

If you don't need a new version or custom build of NVcaffe, you can still use deb packages to install the latest release.
Follow [these instructions](UbuntuInstall.md#repository-access) to gain access to the required repositories, and then use this command to install:
```sh
% sudo apt-get install caffe-nv python-caffe-nv
```

Otherwise, **follow [these instructions](BuildCaffe.md) to build from source**.

## Torch

With v3.0, DIGITS now supports Torch7 as an optional alternative backend to Caffe.

> NOTE: Torch support is still experimental!

As with Caffe, you can use deb packages to install the latest release:
```sh
% sudo apt-get install torch7-nv
```

Otherwise, **follow [these instructions](BuildTorch.md) to build from source**.

# Starting the server

You can run DIGITS in two modes:

### Development mode
```sh
% ./digits-devserver
```

Starts a development server (werkzeug backend) at `http://localhost:5000/`.
```
$ ./digits-devserver --help
usage: digits-devserver [-h] [-p PORT] [-c] [-d] [--version]

Run the DIGITS development server

optional arguments:
  -h, --help            show this help message and exit
  -p PORT, --port PORT  Port to run app on (default 5000)
  -c, --config          Edit the application configuration
  -d, --debug           Run the application in debug mode (reloads when the
                        source changes and gives more detailed error messages)
  --version             Print the version number and exit
```

### Production mode
```sh
% ./digits-server
```

Starts a production server (gunicorn backend) at `http://localhost:34448`.
If you get any errors about an invalid configuration, use the development server first to set your configuration.

If you have installed the nginx.site to `/etc/nginx/sites-enabled/`, then you can view your app at `http://localhost/`.

# Getting started

Now that you're up and running, check out the [getting started guide](GettingStarted.md).

## Troubleshooting

Most configuration options should have appropriate defaults.
If you need to edit your configuration for some reason, try one of these commands:
```sh
# Set options before starting the server
./digits-devserver --config
# Advanced options
python -m digits.config.edit --verbose
```
