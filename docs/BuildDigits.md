# Building DIGITS for Ubuntu Server 16.04

I've built Ubuntu Server instances for DIGITS multiple times and have compiled a list of steps that seems to work well for me. Configuring all the dependencies and installing all the prerequisites can be tricky so I thought I'd share it in one place with some notes.

## Prerequisites

You need an NVIDIA driver to begin with of course ([details and instructions](InstallCuda.md#driver)).

## Install Ubuntu 16.04 64-bit Server
I do the typical installation using LVM to encrypt the whole drive. No automatic updates. Basic installation packages plus SSH server and really nothing else (at first).

### Basic updates/upgrades after installation - modify for your editor of choice
```
sudo apt-get update
sudo apt-get upgrade
sudo apt-get install emacs
```
### I like to try and request a specific IP. Insert the following lines into dhclient.conf. If you can get a static IP even better.
```
emacs /etc/dhcp/dhclient.conf
	#try to get a specific ip; dhclient -r -v to request again if it fails on startup
	send dhcp-requested-address XXX.XX.XXX.XXX;
```
### Install CUDA which will pull the latest Nvidia driver. I have always pulled 8-0 which is good for DIGITS 6.
```
cd /tmp
CUDA_REPO_PKG=http://developer.download.nvidia.com/compute/cuda/repos/ubuntu1604/x86_64/cuda-repo-ubuntu1604_8.0.61-1_amd64.deb
ML_REPO_PKG=http://developer.download.nvidia.com/compute/machine-learning/repos/ubuntu1604/x86_64/nvidia-machine-learning-repo-ubuntu1604_1.0.0-1_amd64.deb
wget "$CUDA_REPO_PKG" -O /tmp/cuda-repo.deb && sudo dpkg -i /tmp/cuda-repo.deb && rm -f /tmp/cuda-repo.deb
wget "$ML_REPO_PKG" -O /tmp/ml-repo.deb && sudo dpkg -i /tmp/ml-repo.deb && rm -f /tmp/ml-repo.deb
sudo apt-get update
sudo apt-get install cuda-8-0
sudo apt-mark hold cuda-8-0
```
### The above, in my experience, pulls the 390 version of the Nvidia drivers. If not, you can do the following.
```
sudo apt-get install nvidia-390
apt-mark hold nvidia-390
```
### Since we are using Nvidia, don't use Nouveau. Make a few changes to not try to boot to the GUI but command line instead.
```
sudo emacs /etc/modprobe.d/blacklist-nouveau.conf
	blacklist nouveau
	blacklist lbm-nouveau
	alias nouveau off
	alias lbm-nouveau off
sudo emacs /etc/default/grub
	GRUB_CMDLINE_LINUX_DEFAULT="text"
	GRUB_CMDLINE_LINUX="text"
sudo systemctl set-default multi-user.target
sudo reboot now
```
### After reboot, double check. The first command should show stuff, the second not (if successful).
```
lsmod | grep nvidia
lsmod | grep nouveau
```

## Dependencies - I choose to put everything in /usr/local but this is up to personal preference.
### Protobuf - in some cases I had trouble with git:// endpoints so I modified to always use https in the second line.
```
sudo apt-get install autoconf automake libtool curl make g++ git python-dev python-setuptools unzip
git config --global url."https://".insteadOf git://
export PROTOBUF_ROOT=/usr/local/protobuf
sudo git clone https://github.com/google/protobuf.git $PROTOBUF_ROOT -b '3.2.x'
cd $PROTOBUF_ROOT
sudo ./autogen.sh
sudo ./configure
sudo make "-j$(nproc)"
sudo make install
sudo ldconfig
cd python
sudo python setup.py install --cpp_implementation
```

### Caffe. I use Cudnn version 5 though I think it will work with version 6. Note that you need to add an environment variable to the .bashrc file for whatever user will be running DIGITS so it knows where Caffe is.
```
sudo apt-get install libcudnn5
sudo apt-mark hold libcudnn5
sudo apt-get install --no-install-recommends build-essential cmake git gfortran libatlas-base-dev libboost-filesystem-dev libboost-python-dev libboost-system-dev libboost-thread-dev libgflags-dev libgoogle-glog-dev libhdf5-serial-dev libleveldb-dev liblmdb-dev libopencv-dev libsnappy-dev python-all-dev python-dev python-h5py python-matplotlib python-numpy python-opencv python-pil python-pip python-pydot python-scipy python-skimage python-sklearn libnccl-dev
emacs ~/.bashrc
	export CAFFE_ROOT=/usr/local/caffe
source ~/.bashrc
sudo git clone https://github.com/NVIDIA/caffe.git $CAFFE_ROOT -b 'caffe-0.15'
pip install -r $CAFFE_ROOT/python/requirements.txt
cd $CAFFE_ROOT
sudo mkdir build
cd build
sudo cmake ..
sudo make -j"$(nproc)"
sudo make install
```

### Torch
```
sudo apt-get install --no-install-recommends git software-properties-common
export TORCH_ROOT=/usr/local/torch
git clone https://github.com/torch/distro.git $TORCH_ROOT --recursive
cd $TORCH_ROOT
sudo ./install-deps
sudo ./install.sh -b
sudo apt-get install --no-install-recommends libhdf5-serial-dev liblmdb-dev
source ~/.bashrc
sudo su
source /usr/local/torch/install/bin/torch-activate
luarocks install tds
luarocks install "https://raw.github.com/deepmind/torch-hdf5/master/hdf5-0-0.rockspec"
luarocks install "https://raw.github.com/Neopallium/lua-pb/master/lua-pb-scm-0.rockspec"
luarocks install lightningmdb 0.9.18.1-1 LMDB_INCDIR=/usr/include LMDB_LIBDIR=/usr/lib/x86_64-linux-gnu
<exit root>
```

### Tensorflow. Using 1.2.1 currently per DIGITS recommendations but I've worked with 1.4 as well. Some of this depends on the compute capability of your card.
```
pip install tensorflow-gpu==1.2.1
```

### DIGITS. Note that I've given control to the local user for the jobs directory and digits log file. This depends on what user will be running the DIGITS server.
```
DIGITS_ROOT=/usr/local/digits
sudo git clone https://github.com/NVIDIA/DIGITS.git $DIGITS_ROOT
pip install -r $DIGITS_ROOT/requirements.txt
sudo pip install -e $DIGITS_ROOT
sudo apt-get install python-tk
sudo mkdir /usr/local/digits/digits/jobs
sudo chown <user>:<user> /usr/local/digits/digits/jobs
sudo touch /usr/local/digits/digits/digits.log
sudo chown <user>:<user> /usr/local/digits/digits/digits.log
```

## Now you should have everything installed and can start DIGITS using the digits-devserver executable. Good idea to test here to be sure. I like to create a startup job as follows.

```
sudo emacs /lib/systemd/system/digits.service
	[Unit]
	Description=Start Digits

	[Service]
	User=<user>
	Environment=CAFFE_ROOT=/usr/local/caffe
	WorkingDirectory=/usr/local/digits
	ExecStart=/bin/bash digits-devserver
	Restart=always
	RestartSec=30

	[Install]
	WantedBy=multi-user.target
systemctl daemon-reload
systemctl enable digits
systemctl start digits
```

# Getting started

Now that you're up and running, check out the [Getting Started Guide](GettingStarted.md).

# Development

If you are interested in developing for DIGITS or work with its source code, check out the [Development Setup Guide](DevelopmentSetup.md)

## Troubleshooting

Most configuration options should have appropriate defaults.
Read [this doc](Configuration.md) for information about how to set a custom configuration for your server.
