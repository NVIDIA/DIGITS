# Building DIGITS

The preferred installation method for DIGITS is using pre-built packages on Ubuntu 14.04 ([instructions](UbuntuInstall.md)).

If those don't work for you for some reason, these instructions will walk you through building the latest version of DIGITS from source.
**These instructions are for installation on Ubuntu 14.04 and 16.04.**

On other platforms, installation of DIGITS itself should be pretty simple since it is a pure Python project.
The difficulty will come from installing and configuring Caffe and/or Torch7.
Though other platforms are not supported officially, users have successfully installed DIGITS and Caffe on CentOS, OSX and even Windows (using the `windows` branch on BVLC/caffe).

## Dependencies

Install some dependencies with Deb packages:
```sh
sudo apt-get install --no-install-recommends git graphviz gunicorn python-dev python-flask python-flaskext.wtf python-gevent python-h5py python-numpy python-pil python-protobuf python-scipy
```

Follow [these instructions](BuildCaffe.md) to build Caffe (**required**).

Follow [these instructions](BuildTorch.md) to build Torch7 (*suggested*).

## Download source

```sh
# example location - can be customized
DIGITS_HOME=~/digits
git clone https://github.com/NVIDIA/DIGITS.git $DIGITS_HOME
```

Throughout the docs, we'll refer to your install location as `DIGITS_HOME` (`~/digits` in this case), though you don't need to actually set that environment variable.

## Python packages

Several PyPI packages need to be installed:
```sh
sudo pip install -r $DIGITS_HOME/requirements.txt
```

# Starting the server

You can run DIGITS in two modes:

### Development mode

```sh
./digits-devserver
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
./digits-server
```

Starts a production server (gunicorn backend) at `http://localhost:34448`.
If you get any errors about an invalid configuration, use the development server first to set your configuration.

If you have installed the nginx.site to `/etc/nginx/sites-enabled/`, then you can view your app at `http://localhost/`.

# Getting started

Now that you're up and running, check out the [Getting Started Guide](GettingStarted.md).

## Troubleshooting

Most configuration options should have appropriate defaults.
If you need to edit your configuration for some reason, try one of these commands:
```sh
# Set options before starting the server
./digits-devserver --config
# Advanced options
python -m digits.config.edit --verbose
```
