# Image Folder Specification

DIGITS can train models which classify images into different categories. This document explains how to structure a folder used as input for creating a dataset. You can also create textfiles to specify the images manually, but using a folder structure is recommended. This is a simple and clean way to layout your data.

## Format

* Create a folder (e.g. `my-images`).
* For each category, create a folder within that folder (e.g. `my-images/cat`).
* Fill each folder with pictures that you want to use for that category.
```
my-images/
├── cat/
│   ├── 1.jpg
│   └── 2.jpg
└── dog/
    ├── 1.jpg
    └── 2.jpg
```

Then use `/path/to/my-images` for your "Training Images" folder. It's as simple as that!

### Filenames

Currently, only 4 file formats are supported: PNG, JPEG, BMP and PPM. If you would like for DIGITS to support another file format, just ask!

File extensions are case-insensitive and the format of your file name is not important. All of the following filenames are valid:
```
my-images/
├── cat/
│   ├── 1.PNG
│   ├── 2.JPEG
│   └── lolcat.jpg
└── dog/
    ├── 000000001.jpg
    ├── 2.BMP
    └── big-dog.png
```

### Subfolders

Any subfolders within a category folder are essentially flattened. All files found in a recursive search are added to the dataset.
```
my-images/
├── cat/
│   ├── other-cat.jpg
│   ├── siamese/
│   │   └── 1.jpg
│   └── tabby/
│       └── 1.jpg
└── dog/
    ├── big/
    │   ├── great-dane/
    │   │   └── 1.jpg
    │   └── other-big-dog.jpg
    ├── other-dog.jpg
    └── small/
        └── jack_russell/
            └── 1.jpg
```

In this case, DIGITS will find 7 images in 2 categories:
```
<cat>
my-images/cat/siamese/1.jpg
my-images/cat/other-cat.jpg
my-images/cat/tabby/1.jpg

<dog>
my-images/dog/big/great-dane/1.jpg
my-images/dog/big/other-big-dog.jpg
my-images/dog/other-dog.jpg
my-images/dog/small/jack_russell/1.jpg
```

### Multiple top-level folders

If you have all of your data in one top-level folder, DIGITS can automatically split up your data for you. For example, if you have 6 cat images and 8 dog images, and you choose a 50/50 split between training and validation images, then you will have 3 cat and 4 dog images randomly selected for your training set, and the rest will be used in the validation set.

But, if you need more control and you want to select which images should be used, then you can create another top-level folder, provided the categories match:
```
.
├── train-images/
│   ├── cat/
│   │   └── 1.jpg
│   └── dog/
│       └── 1.jpg
└── val-images/
    ├── cat/
    │   └── 1.jpg
    └── dog/
        └── 1.jpg
```

Then, you can use `train-images` for your training images and `val-images` as a separate folder for validation images.

## Example use case - ImageNet subset

As an example, these features can be combined to create small groupings of ImageNet categories for classification. In the example folder below, symlinks are used to avoid data replication on disk. First, the validation images need to be sorted by category folders as the training images already are. Then you can do something like this:
```
imagenet_subset1/
├── train
│   ├── bear
│   │   ├── n02132136 -> /data/images/imagenet12/train/n02132136
│   │   ├── n02133161 -> /data/images/imagenet12/train/n02133161
│   │   └── n02134084 -> /data/images/imagenet12/train/n02134084
│   ├── dog
│   │   ├── n02099601 -> /data/images/imagenet12/train/n02099601
│   │   ├── n02099712 -> /data/images/imagenet12/train/n02099712
│   │   ├── n02101388 -> /data/images/imagenet12/train/n02101388
│   │   ├── n02106166 -> /data/images/imagenet12/train/n02106166
│   │   └── n02106662 -> /data/images/imagenet12/train/n02106662
│   ├── fish
│   │   ├── n01440764 -> /data/images/imagenet12/train/n01440764
│   │   ├── n01443537 -> /data/images/imagenet12/train/n01443537
│   │   ├── n02536864 -> /data/images/imagenet12/train/n02536864
│   │   └── n02640242 -> /data/images/imagenet12/train/n02640242
│   ├── monkey
│   │   ├── n02486410 -> /data/images/imagenet12/train/n02486410
│   │   ├── n02490219 -> /data/images/imagenet12/train/n02490219
│   │   ├── n02492660 -> /data/images/imagenet12/train/n02492660
│   │   └── n02493793 -> /data/images/imagenet12/train/n02493793
│   ├── snake
│   │   ├── n01729977 -> /data/images/imagenet12/train/n01729977
│   │   ├── n01734418 -> /data/images/imagenet12/train/n01734418
│   │   └── n01735189 -> /data/images/imagenet12/train/n01735189
└── val
    ├── bear
    │   ├── n02132136 -> /data/images/imagenet12/val/n02132136
    │   ├── n02133161 -> /data/images/imagenet12/val/n02133161
    │   └── n02134084 -> /data/images/imagenet12/val/n02134084
    ├── dog
    │   ├── n02099601 -> /data/images/imagenet12/val/n02099601
    │   ├── n02099712 -> /data/images/imagenet12/val/n02099712
    │   ├── n02101388 -> /data/images/imagenet12/val/n02101388
    │   ├── n02106166 -> /data/images/imagenet12/val/n02106166
    │   └── n02106662 -> /data/images/imagenet12/val/n02106662
    ├── fish
    │   ├── n01440764 -> /data/images/imagenet12/val/n01440764
    │   ├── n01443537 -> /data/images/imagenet12/val/n01443537
    │   ├── n02536864 -> /data/images/imagenet12/val/n02536864
    │   └── n02640242 -> /data/images/imagenet12/val/n02640242
    ├── monkey
    │   ├── n02486410 -> /data/images/imagenet12/val/n02486410
    │   ├── n02490219 -> /data/images/imagenet12/val/n02490219
    │   ├── n02492660 -> /data/images/imagenet12/val/n02492660
    │   └── n02493793 -> /data/images/imagenet12/val/n02493793
    ├── snake
    │   ├── n01729977 -> /data/images/imagenet12/val/n01729977
    │   ├── n01734418 -> /data/images/imagenet12/val/n01734418
    │   └── n01735189 -> /data/images/imagenet12/val/n01735189
```
