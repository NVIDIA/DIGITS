# Installing TensorFlow

DIGITS now supports TensorFlow as an optional alternative backend to Caffe or Torch.

Installation for [Ubuntu](https://www.tensorflow.org/install/install_linux#installing_with_virtualenv)

Installation for [Mac](https://www.tensorflow.org/install/install_mac#installing_with_virtualenv)

## Requirements

DIGITS is current targeting tensorflow-gpu V1.2.1.

TensorFlow for DIGITS requires one or more NVIDIA GPUs with CUDA Compute Capbility of 3.0 or higher. See [the official GPU support list](https://developer.nvidia.com/cuda-gpus) to see if your GPU supports it.

Along with that requirement, the following should be installed

* One or more NVIDIA GPUs ([details](InstallCuda.md#gpu))
* An NVIDIA driver ([details and installation instructions](InstallCuda.md#driver))
* A CUDA toolkit ([details and installation instructions](InstallCuda.md#cuda-toolkit))
* cuDNN 5.1 ([download page](https://developer.nvidia.com/cudnn))

### A Note About cuDNN and TensorFlow
Currently tensorflow v1.2.1 targets cuDNN 5.1. **To have tensorflow v1.2.1 running in digits, you must have cuDNN 5.1 installed.** To install it, use the following command in a terminal

```
sudo apt-get install libcudnn5
```


## Installation

These instructions are based on [the official TensorFlow instructions]
(https://www.tensorflow.org/versions/master/install/)

TensorFlow comes with pip, to install it, just simply use the command
```
pip install tensorflow-gpu==1.2.1
```

TensorFlow should then install effortlessly and pull in all its required dependices.

## Getting Started With TensorFlow In DIGITS

Follow [these instructions](GettingStartedTensorflow.md) for information on getting started with TensorFlow in DIGITS
