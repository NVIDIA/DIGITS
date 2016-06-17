# CUDA Installation

Getting CUDA and the NVIDIA driver installed correctly on your machine can be tricky.
This guide should tell you everything you need to know (on Ubuntu, at least!).

Another good resource is the [CUDA installation guide for Linux](http://docs.nvidia.com/cuda/cuda-installation-guide-linux).

## GPU

You will need an NVIDIA GPU if you want to use CUDA.
**If you want to use [cuDNN](https://developer.nvidia.com/cudnn), you will need a GPU with compute capability >= 3.0.**
To find out what the compute capability of your card is, check one of these websites:

* https://developer.nvidia.com/cuda-gpus
* https://en.wikipedia.org/wiki/CUDA#GPUs_supported

You can also use the DIGITS `device_query` tool to check for the compute `major` and `minor` versions:
```sh
$ digits/device_query.py
Device #0:
>>> CUDA attributes:
  name                         Tesla K40c
  totalGlobalMem               12079136768
  clockRate                    745000
  major                        3
  minor                        5
>>> NVML attributes:
  Total memory                 11519 MB
  Used memory                  23 MB
  Memory utilization           0%
  GPU utilization              0%
  Temperature                  30 C
```

## Driver

On Ubuntu, you can install a driver in two ways: with a `run` file or with a Deb package.

It is recommended that you use a Deb package to install your driver, unless you have a new GPU that requires a newer driver version.
Deb packages are simpler to install, uninstall and upgrade, while `run` file installers are useful if you need a newer driver version.

To install with a `run` file, download one from the [NVIDIA Driver Downloads](http://www.nvidia.com/Download/index.aspx) website and follow the instructions.
If you run into any problems, look at the "Additional Information" section.

> **IMPORTANT**: If you use a `run` file to install your driver, don't install the `cuda` Deb package.
More information [below](#deb-packages).

## CUDA Toolkit

On the [CUDA Downloads](https://developer.nvidia.com/cuda-downloads) website, you will see three options for installing the toolkit: `runfile (local)`, `deb (local)`, `deb (network)`.

1. `deb (network)` - This is a Deb package, the preferred method.
This gives you access to all of the packages in the [CUDA repository](http://developer.download.nvidia.com/compute/cuda/repos/ubuntu1404/x86_64/), including multiple toolkit versions.

    ```sh
    dpkg -i cuda-repo-ubuntu1404_7.5-18_amd64.deb
    apt-get update
    # Don't run this command yet - read below first
    #    apt-get install cuda
    ```

2. `deb (local)` - Also a Deb package, a nice option if you have a bad network connection.
The downside is that you can't get package updates and you have to install separate packages for CUDA 7.0, 7.5, etc.

    ```sh
    dpkg -i cuda-repo-ubuntu1404-7-5-local_7.5-18_amd64.deb
    apt-get update
    # Don't run this command yet - read below first
    #    apt-get install cuda
    ```

3. `runfile (local)` - Shell script.  Don't use this unless you have to for some reason.
As with the driver ([see above](#driver)), it is more difficult to uninstall or upgrade your CUDA installation if you use a `run` file  installer.

    ```sh
    sh cuda_7.5.18_linux.run
    ```

### Deb packages

Assuming you chose to use a Deb package above, here are some of the packages you can install:

1. `apt-get install cuda` - This will install the latest toolkit (currently 7.5) and the latest driver (currently `nvidia-352`).
  * **IMPORTANT**: Don't install this package if you installed your driver with a `run` file. The Deb package may not be able to fully uninstall your `run` file driver installation.
2. `apt-get install cuda-toolkit-7-5` - Installs only the toolkit and not the driver.
3. `apt-get install cuda-drivers` - Installs only the driver and not the toolkit.

For more information, see [the "Meta Packages" section](http://docs.nvidia.com/cuda/cuda-installation-guide-linux/#package-manager-metas) of the CUDA installation guide for Linux.

## Environment

> NOTE: Your environment will be set up automatically with the CUDA 8.0 installers.

Finally, you need to set up your environment correctly so that the runtime linker can find your shared libraries.
There are a few ways to do this:

1. Add an entry to `/etc/ld.so.conf.d/`.
  * This requires `sudo` privileges.

    ```
    echo "/usr/local/cuda/lib64" | sudo tee /etc/ld.so.conf.d/cuda64.conf
    sudo ldconfig
    ```

2. Edit `LD_LIBRARY_PATH`.
  * This does not require `sudo` privileges.
  * The exact formula required depends on which shell you are using and how you login to your machine.

    ```sh
    # Login shell
    echo "export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/cuda/lib64" >> ~/.profile && source ~/.profile

    # Non-login interactive shell (bash)
    echo "export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/cuda/lib64" >> ~/.bashrc && source ~/.bashrc
    ```

  * Further reading on setting persistent environment variables:
    * http://unix.stackexchange.com/q/117467/99570
    * http://askubuntu.com/q/210884/336440

3. Install the `cuda-ld-conf-7-0` package
  * This package is made available on NVIDIA's [machine learning repo](http://developer.download.nvidia.com/compute/machine-learning/repos/ubuntu1404/x86_64).
  * When you install DIGITS with a Deb package, this package gets installed automatically, so you don't have to worry about setting up your environment.
  * It simply sets up option (1) for you automatically.
