# Standard Datasets

Table of Contents
=================
* [Overview](#overview)
* Details
  * [MNIST](#mnist)
  * [CIFAR](#cifar)
      * [CIFAR10](#cifar10)
      * [CIFAR100](#cifar100)

## Overview

DIGITS will download some standard datasets for you and store them for you locally in the format that DIGITS expects (see [Image Folder Format](ImageFolderFormat.md) for a detailed explanation). Once these folders are created, you can use them to create your datasets with DIGITS.

![HTML Form](images/standard-datasets/html-form.jpg)

```
$ python -m digits.download_data -h
usage: __main__.py [-h] [-c] dataset output_dir

Download-Data tool - DIGITS

positional arguments:
  dataset      mnist/cifar10/cifar100
  output_dir   The output directory for the data

optional arguments:
  -h, --help   show this help message and exit
  -c, --clean  Clean out the directory first (if necessary)
```

## MNIST

Yann LeCun provides a dataset of 28x28 grayscale images of handwritten digits. You can read all about it here:
http://yann.lecun.com/exdb/mnist/

> The MNIST database of handwritten digits, available from this page, has a training set of 60,000 examples, and a test set of 10,000 examples. It is a subset of a larger set available from NIST. The digits have been size-normalized and centered in a fixed-size image.
>
> It is a good database for people who want to try learning techniques and pattern recognition methods on real-world data while spending minimal efforts on preprocessing and formatting.

Run this:
```sh
$ python -m digits.download_data mnist ~/mnist
```

And these folders and files will be created for you (images and temporary files omitted):
```
mnist/
├── train/
│   ├── 0/
│   ├── 1/
│   ├── 2/
│   ├── 3/
│   ├── 4/
│   ├── 5/
│   ├── 6/
│   ├── 7/
│   ├── 8/
│   ├── 9/
│   ├── labels.txt
│   └── train.txt
└── test/
    ├── 0/
    ├── ...
    ├── 9/
    ├── labels.txt
    └── test.txt
```

Then, you can use `~/mnist/train` for your training images and `~/mnist/test` for your validation or test images.

## CIFAR

Alex Krizhevsky provides two datasets of 32x32 color images. You can read all about them here:
http://www.cs.toronto.edu/~kriz/cifar.html

### CIFAR10

> The CIFAR-10 dataset consists of 60000 32x32 colour images in 10 classes, with 6000 images per class. There are 50000 training images and 10000 test images.

Run this:
```sh
$ python -m digits.download_data cifar10 ~/cifar10
```
And these folders and files will be created for you (images and temporary files omitted):
```
cifar10
├── train/
│   ├── airplane/
│   ├── automobile/
│   ├── bird/
│   ├── cat/
│   ├── deer/
│   ├── dog/
│   ├── frog/
│   ├── horse/
│   ├── ship/
│   ├── truck/
│   ├── labels.txt
│   └── train.txt
└── test/
    ├── airplane/
    ├── ...
    ├── truck/
    ├── labels.txt
    └── test.txt
```

Then, you can use `~/cifar10/train` for your training images and `~/cifar10/test` for your validation or test images.

### CIFAR100

> This dataset is just like the CIFAR-10, except it has 100 classes containing 600 images each. There are 500 training images and 100 testing images per class. The 100 classes in the CIFAR-100 are grouped into 20 superclasses. Each image comes with a "fine" label (the class to which it belongs) and a "coarse" label (the superclass to which it belongs).

Run this:
```sh
$ python -m digits.download_data cifar100 ~/cifar100
```
And these folders and files will be created for you (images and temporary files omitted):
```
cifar100/
├── coarse/
│   ├── train/
│   │   └── ...
│   ├── test/
│   │   └── ...
│   ├── labels.txt
│   ├── test.txt
│   └── train.txt
└── fine/
    ├── train/
    │   └── ...
    ├── test/
    │   └── ...
    ├── labels.txt
    ├── test.txt
    └── train.txt
```

If you want to use the coarse dataset (10 classes), use `~/cifar100/coarse/train` and `~/cifar100/coarse/test`.

If you want to use the fine dataset (100 classes), use `~/cifar100/fine/train` and `~/cifar100/fine/test`.
