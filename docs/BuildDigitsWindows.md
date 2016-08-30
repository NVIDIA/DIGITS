# Build DIGITS on Windows

## Limitation
DIGITS for Windows depends on Windows branch of BVLC Caffe.
The following layers, required for DetectNet feature, are not implemented in that branch.
- detectnet_transform_layer
- l1_loss_layer

As a result, DIGITS for Windows does not support DetectNet.
To run DIGITS with DetectNet, please use NV-Caffe 0.15 or above on Ubuntu.


## Prerequisites
- Python2
- CUDA 7.5
- CuDNN 5.1
- Caffe
- Graphviz



## Installing prerequisites

### Python2
Download and install Python 2.7.11 64bit from Python's official site (https://www.python.org/ftp/python/2.7.11/python-2.7.11.amd64.msi).
Please select Add Python Path during installation.

Download numpy, scipy, matplotlib, scikit-image, h5py from Unofficial Windows Binaries for Python Extension Packages webpage at (http://www.lfd.uci.edu/~gohlke/pythonlibs/).
Remember to download correct version (2.7) and architecture (64-bit).

Additionally, download gevent v1.0.2 at the same site.
Run command prompt (cmd.exe) as administrator, and issue the following commands.
```
python -m pip install cython
python -m pip install numpy-1.11.0+mkl-cp27-cp27m-win_amd64.whl
python -m pip install scipy-0.17.0-cp27-none-win_amd64.whl
python -m pip install matplotlib-1.5.1-cp27-none-win_amd64.whl
python -m pip install scikit_image-0.12.3-cp27-cp27m-win_amd64.whl
python -m pip install h5py-2.6.0-cp27-cp27m-win_amd64.whl
```

If the installation process complains compiler not found, you need to install Microsoft Visual C++ Compiler for Python 2.7, downloaded at (https://www.microsoft.com/en-us/download/details.aspx?id=44266).
We recommend installing it by
```
msiexec /i VCForPython27.msi ALLUSERS=1
```

After that compiler is installed, finish the above python -m pip install commands.

At this moment, do not install gevent yet.  We need to install it after installing DIGITS.

### CUDA 7.5
CUDA 7.5 can be obtained at NVIDIA CUDA (https://developer.nvidia.com/cuda-downloads).
Please select Windows 7 to download.

### CuDNN 5.1
Download CuDNN 5.1 at NVIDIA website (https://developer.nvidia.com/cudnn).
Please select CuDNN 5.1 for CUDA 7.5.

### Caffe
Caffe can be obtained at (https://github.com/bvlc/caffe/tree/windows).
Note you need to install Visual Studio 2013 to build Caffe.
Before building it, enable Python support, CUDA and CuDNN by following instructions on the same page.
Because we are using Official CPython, please change the value of PythonDir tag from C:\Miniconda2\ to C:\PYTHON27\ (assume your CPython installation is the default C:\PYTHON27\).
After building it, configure your Python environment to include pycaffe, which is described at (https://github.com/bvlc/caffe/tree/windows#remark).
Your caffe.exe will be inside Build\x64\Release  directory (if you made release build).

### Graphviz
Graphviz is available at (www.graphviz.org/Download.php).
Please note this site is not always available online.
The installation directory can not contain space, so don't install it under the regular 'c:\Program Files (x86)' directory.
Try something like 'c:\graphviz' instead.
When the installation directory contains space, pydot could not launch the dot.exe file, even it has no problem finding it.
Add the c:\graphviz\bin directory to your PATH.  

## Installing DIGITS

Clone DIGITS from github.com (https://github.com/nvidia/digits).
From the command prompt (run as administrator) and cd to DIGITS directory.
Then type
```
python -m pip install -r requirements.txt
```

You may see error about Pillow, like
``` ValueError: jpeg is required unless explicitly disabled using --disable-jpeg, aborting ```
If this happens, download Pillow Windows Installer (Pillow-3.1.1.win-amd64-py2.7.exe) at https://pypi.python.org/pypi/Pillow/3.1.1 and run the exectuables.
After installing Pillow in the above way, run
```
python -m pip install -r requirements.txt
```
again.

After the above command, check if all required Python dependencies are met by comparing requirements.txt and output of the following command.
```
python -m pip list
```

If gevent is not v1.0.2, install it from the whl file, downloaded previously from (http://www.lfd.uci.edu/~gohlke/pythonlibs/).
```
python -m pip install gevent-1.0.2-cp27-none-win_amd64.whl
```

It should uninstall the gevent you had, and install gevent 1.0.2.

Because readline is not available in Windows, you need to install one additional Python package.
```
python -m pip install pyreadline
```

 
## Running DIGITS

First, check if caffe executable is included in your PATH environment variable.
If not, add it.
```
set PATH=%PATH%;MY_CAFFE_ROOT\Build\x64\Release
```
Replace MY_CAFFE_ROOT with your local caffe directory.

Launch DIGITS devserver with the following command:
```
python digits-devserver
```
Point your browser to localhost:5000.  You should be able to see DIGITS.


## Troubleshooting

### DIGITS crashes when trying to classify images with ** Show visualizations and statistics **

This issue should have been resolved.
However, if you still encounter this issue, this seems related to different hdf5 DLL binding between pycaffe and h5py.
The DLL used by pycaffe was pulled from nuget, and its version is 1.8.15.2.
Slightly older than the DLL in h5py.
A temporary solution is to load h5py before pycaffe.
To force loading h5py before pycaffe, you can either add one line at the beginning of digits-devserver file, or import h5py just before import caffe in digits/config/caffe_option.py.

### import readline causes ImportError

Change import readline in digits\config\prompt.py to
```py
try:
    import readline
except ImportError:
    import pyreadline as readline
```

### DIGITS complains Torch binary not found in PATH

Currently, DIGITS does not support Torch on Windows platform.

