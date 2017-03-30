# DIGITS

[![Build Status](https://travis-ci.org/NVIDIA/DIGITS.svg?branch=master)](https://travis-ci.org/NVIDIA/DIGITS)

DIGITS (the **D**eep Learning **G**PU **T**raining **S**ystem) is a webapp for training deep learning models.

# Installation

| Installation method | Supported platform[s] | Available versions | Instructions |
| --- | --- | --- | --- |
| Deb packages | Ubuntu 14.04, 16.04 | [14.04 repo](http://developer.download.nvidia.com/compute/machine-learning/repos/ubuntu1404/x86_64), [16.04 repo](http://developer.download.nvidia.com/compute/machine-learning/repos/ubuntu1604/x86_64) | [docs/UbuntuInstall.md](docs/UbuntuInstall.md) |
| Docker | Linux | [DockerHub tags](https://hub.docker.com/r/nvidia/digits/tags/) | [nvidia-docker wiki](https://github.com/NVIDIA/nvidia-docker/wiki/DIGITS) |
| Source | Ubuntu 14.04, 16.04 | [GitHub tags](https://github.com/NVIDIA/DIGITS/releases) | [docs/BuildDigits.md](docs/BuildDigits.md) |

# Usage

Once you have installed DIGITS, visit [docs/GettingStarted.md](docs/GettingStarted.md) for an introductory walkthrough.

Then, take a look at some of the other documentation at [docs/](docs/) and [examples/](examples/):

* [Getting started with Torch](docs/GettingStartedTorch.md)
* [Fine-tune a pretrained model](examples/fine-tuning/README.md)
* [Train an autoencoder network](examples/autoencoder/README.md)
* [Train a regression network](examples/regression/README.md)
* [Train a Siamese network](examples/siamese/README.md)
* [Train a text classification network](examples/text-classification/README.md)
* [Train an object detection network](examples/object-detection/README.md)
* [Learn more about weight initialization](examples/weight-init/README.md)
* [Use Python layers in your Caffe networks](examples/python-layer/README.md)
* [Download a model and use it to classify an image outside of DIGITS](examples/classification/README.md)
* [Overview of the REST API](docs/API.md)

# Get help

### Installation issues
* First, check out the instructions above
* Then, ask questions on our [user group](https://groups.google.com/d/forum/digits-users)

### Usage questions
* First, check out the [Getting Started](docs/GettingStarted.md) page
* Then, ask questions on our [user group](https://groups.google.com/d/forum/digits-users)

### Bugs and feature requests
* Please let us know by [filing a new issue](https://github.com/NVIDIA/DIGITS/issues/new)
* Bonus points if you want to contribute by opening a [pull request](https://help.github.com/articles/using-pull-requests/)!
  * You will need to send a signed copy of the [Contributor License Agreement](CLA) to digits@nvidia.com before your change can be accepted.

