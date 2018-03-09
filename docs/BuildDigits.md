# Building DIGITS for Ubuntu Server 16.04

I've built Ubuntu Server instances for DIGITS multiple times and have compiled a list of steps that seems to work well for me. Configuring all the dependencies and installing all the prerequisites can be tricky so I thought I'd share it in one place with some notes. It does depend somewhat on the compute capability of your Nvidia card.

## Prerequisites

You need an NVIDIA driver to begin with of course ([details and instructions](InstallCuda.md#driver)).

## Install Ubuntu 16.04 64-bit Server
I do the typical installation using LVM to encrypt the whole drive. No automatic updates. Basic installation packages plus SSH server and really nothing else (at first).

### Basic updates/upgrades after installation - modify for your editor of choice
```
sudo apt-get update
sudo apt-get upgrade
sudo apt-get install emacs    #sorry vim fanboys
```

### I like to try and request a specific IP. Insert the following lines into dhclient.conf. If you can get a static IP even better.
```
sudo emacs /etc/dhcp/dhclient.conf
	#try to get a specific ip; dhclient -r -v to request again if it fails on startup
	send dhcp-requested-address XXX.XX.XXX.XXX;
```

### Install CUDA which will pull the latest Nvidia driver. I am currently using 9.0 with DIGITS 6.1.
```
cd /tmp
sudo apt-key adv --fetch-keys http://developer.download.nvidia.com/compute/cuda/repos/ubuntu1604/x86_64/7fa2af80.pub
CUDA_REPO_PKG=http://developer.download.nvidia.com/compute/cuda/repos/ubuntu1604/x86_64/cuda-repo-ubuntu1604_9.0.176-1_amd64.deb
ML_REPO_PKG=http://developer.download.nvidia.com/compute/machine-learning/repos/ubuntu1604/x86_64/nvidia-machine-learning-repo-ubuntu1604_1.0.0-1_amd64.deb
wget "$CUDA_REPO_PKG" -O /tmp/cuda-repo.deb && sudo dpkg -i /tmp/cuda-repo.deb && rm -f /tmp/cuda-repo.deb
wget "$ML_REPO_PKG" -O /tmp/ml-repo.deb && sudo dpkg -i /tmp/ml-repo.deb && rm -f /tmp/ml-repo.deb
sudo apt-get update
sudo apt-get install cuda-9-0
sudo apt-mark hold cuda-9-0
wget https://developer.nvidia.com/compute/cuda/9.0/Prod/patches/2/cuda-repo-ubuntu1604-9-0-local-cublas-performance-update-2_1.0-1_amd64-deb
sudo dpkg -i cuda-repo-ubuntu1604-9-0-local-cublas-performance-update-2_1.0-1_amd64-deb
sudo apt-mark hold nvidia-390
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
sudo -H git clone https://github.com/google/protobuf.git $PROTOBUF_ROOT
cd $PROTOBUF_ROOT
sudo -H ./autogen.sh
sudo -H ./configure
sudo -H make "-j$(nproc)"
sudo -H make install
sudo -H ldconfig
cd python
sudo -H python setup.py install --cpp_implementation
```

### Caffe. Update: the ubuntu libcudnn libraries are not found when you try and build Caffe. It's possible you could set environment variables to find them, but now I'm building from scratch using cudnn v. 7.0.5 for cuda 9.0 as follows. I'm also now pulling the latest version of caffe (curently 0.16). Note that you need to add an environment variable to the .bashrc file for whatever user will be running DIGITS so it knows where Caffe is.
```
# download ubuntu cudnn packages from https://developer.nvidia.com/rdp/cudnn-download to match your version of CUDA and Ubuntu
sudo dpkg -i libcudnn7_7.0.5.15-1+cuda9.0_amd64.deb
sudo dpkg -i libcudnn7-dev_7.0.5.15-1+cuda9.0_amd64.deb
sudo dpkg -i libcudnn7-doc_7.0.5.15-1+cuda9.0_amd64.deb
# test your cudnn installation
cd /tmp
cp -r /usr/src/cudnn_samples_v7/ .
cd cudnn_samples_v7/mnistCUDNN/
make clean
make
./mnistCUDNN   #should say "Test Passed!"
rm -rf /tmp/cudnn_samples_v7
sudo -H apt-get install --no-install-recommends build-essential cmake git gfortran libatlas-base-dev libboost-filesystem-dev libboost-python-dev libboost-system-dev libboost-thread-dev libgflags-dev libgoogle-glog-dev libhdf5-serial-dev libleveldb-dev liblmdb-dev libopencv-dev libsnappy-dev python-all-dev python-dev libnccl-dev libboost-regex-dev libturbojpeg libopenblas-dev
emacs ~/.bashrc
	export CAFFE_ROOT=/usr/local/caffe
source ~/.bashrc   #you may need to source or log out and log back in again to get pip added to your PATH
sudo -H git clone https://github.com/NVIDIA/caffe.git $CAFFE_ROOT
sudo -H pip install -r $CAFFE_ROOT/python/requirements.txt   #you seem to need root access at least for some of these libraries
cd $CAFFE_ROOT
sudo -H mkdir build
cd build
sudo -H cmake ..
sudo -H make -j"$(nproc)"
sudo -H make install
```

### Torch
```
sudo apt-get install --no-install-recommends git software-properties-common
export TORCH_ROOT=/usr/local/torch
sudo git clone https://github.com/torch/distro.git $TORCH_ROOT --recursive
cd $TORCH_ROOT
sudo su
./install-deps
export TORCH_NVCC_FLAGS="-D__CUDA_NO_HALF_OPERATORS__"   #something with cuda 9.0? needs to be in root environment
./install.sh -b
exit  #exit root
sudo apt-get install --no-install-recommends libhdf5-serial-dev liblmdb-dev
source ~/.bashrc
sudo su
source /usr/local/torch/install/bin/torch-activate
git config --global url."https://".insteadOf git://   #because you need to do this as root as well for the following
luarocks install tds
luarocks install "https://raw.github.com/deepmind/torch-hdf5/master/hdf5-0-0.rockspec"
luarocks install "https://raw.github.com/Neopallium/lua-pb/master/lua-pb-scm-0.rockspec"
luarocks install lightningmdb 0.9.18.1-1 LMDB_INCDIR=/usr/include LMDB_LIBDIR=/usr/lib/x86_64-linux-gnu
exit  #exit root
```

### Tensorflow. Using 1.5 currently. Some of this depends on the compute capability of your card - I think must be at least 3.5 for tensorflow 1.5.
```
pip install tensorflow-gpu==1.5
#test tensorflow if you want
python
    import tensorflow as tf
    hello = tf.constant('Hello, TensorFlow!')
    sess = tf.Session()
    print(sess.run(hello))    #should get "Hello, TensorFlow!"
```

### DIGITS. Note that I've given control to the local user for the jobs directory and digits log file. This depends on what user will be running the DIGITS server.
```
DIGITS_ROOT=/usr/local/digits
sudo git clone https://github.com/NVIDIA/DIGITS.git $DIGITS_ROOT
sudo -H pip install -r $DIGITS_ROOT/requirements.txt
sudo -H pip install -e $DIGITS_ROOT
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
