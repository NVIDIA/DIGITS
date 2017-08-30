# Using DIGITS to train an Object Detection network

Table of Contents
=================
* [Introduction](#introduction)
* [Dataset creation](#dataset-creation)
    * [Downloading and preparing the KITTI data](#downloading-and-preparing-the-kitti-data)
    * [Loading the data into DIGITS](#loading-the-data-into-digits)
* [Model creation](#model-creation)
    * [DetectNet](#detectnet)
    * [Training DetectNet in DIGITS](#training-detectnet-in-digits)
* [Verification](#verification)

## Introduction

In this tutorial we will see how DIGITS may be used to train an Object Detection neural network using the Caffe back-end.
In this particular example, we will train the network to detect cars in pictures taken from a dashboard camera.
During inference, object detection will be materialized by drawing bounding rectangles around the detected objects.

## Dataset creation

In this example, we will be using data from the Object Detection track of the KITTI Vision Benchmark Suite.
You can of course use any other data you like, but DIGITS expects object detection data to be labelled in the style of KITTI data.

If you do want to use your own dataset instead of KITTI, read [digits/extensions/data/objectDetection/README.md](../../digits/extensions/data/objectDetection/README.md) to format your data properly and then skip the next section.

### Downloading and preparing the KITTI data

We are unable to provide download links to the KITTI data like we can for MNIST and CIFAR, so you'll have to download a few large files yourself.
Go to http://www.cvlibs.net/datasets/kitti/eval_object.php and download these files:

Description | Filename | Size
------------ | ------------- | -------------
Left color images of object data set | `data_object_image_2.zip` | **12GB**
Training labels of object data set | `data_object_label_2.zip` | 5MB
Object development kit | `devkit_object.zip` | 1MB

Copy those files into `$DIGITS_ROOT/examples/object-detection/`.

Then, use the `prepare_kitti_data.py` script to create a train/val split of the labelled images.
This will take a few minutes, spent mostly on unpacking the large zipfiles.
```
$ ./prepare_kitti_data.py
Extracting zipfiles ...
Unzipping data_object_label_2.zip ...
Unzipping data_object_image_2.zip ...
Unzipping devkit_object.zip ...
Calculating image to video mapping ...
Splitting images by video ...
Creating train/val split ...
Done.
```

At the end you will have your data at `$DIGITS_ROOT/examples/object-detection/kitti-data/{train,val}/`.

The data is structured in the following way:
- An image folder containing supported images (`.png`, `.jpg`, etc.).
- A label folder containing `.txt` files in KITTI format that define the ground truth.
For each image in the image folder there must be a corresponding text file in the label folder.
For example if the image folder includes an image named `foo.png` then the label folder needs to include a file named `foo.txt`.

### Loading the data into DIGITS

On the DIGITS home page, select the `Datasets` tab then click `New Dataset > Images > Object Detection`:

![select dataset](select-object-detection-dataset.jpg)

On the dataset creation page, specify the paths to the image and label folders for each of the training and validation sets.
Other fields can be left to their default value.
Finally, give your dataset a name and click `Create`:

![dataset form](form-object-detection-dataset.jpg)

After you have created your dataset you may review data properties by visiting the dataset page.
In the below example there are 5984 images in the training set and 1496 images in the validation set:

![dataset form](dataset-review.jpg)

## Model creation

### DetectNet

In this example we will use **DetectNet**.
DetectNet is a GoogLeNet-derived network that is specifically tuned for Object Detection.

For more information on DetectNet, please refer to [this blog post](https://devblogs.nvidia.com/parallelforall/detectnet-deep-neural-network-object-detection-digits/).

In order to train DetectNet, [NVcaffe](https://github.com/NVIDIA/caffe) version [0.15.1](https://github.com/NVIDIA/caffe/tree/v0.15.1) or later is required.
The model description for DetectNet can be found at `$CAFFE_ROOT/examples/kitti/detectnet_network.prototxt` ([raw link](https://raw.githubusercontent.com/NVIDIA/caffe/caffe-0.15/examples/kitti/detectnet_network.prototxt)).

Since DetectNet is derived from GoogLeNet it is strongly recommended to use pre-trained weights from an ImageNet-trained GoogLeNet as this will help speed training up significantly.
A suitable pre-trained GoogLeNet `.caffemodel` may be found on this [page](https://github.com/BVLC/caffe/tree/rc3/models/bvlc_googlenet).

### Training DetectNet in DIGITS

On the DIGITS home page, select the `Models` tab then click `New Model > Images > Object Detection`:

![select dataset](select-object-detection-model.jpg)

On the model creation page:
- Select the dataset that was created in the previous section.
- Set `Subtract mean` to `None`.
- Set the base learning rate to 0.0001.
- Select the `ADAM` solver.
- Select the `Custom Network` tab.
  - Make sure the `Caffe` sub-tab is selected.
  - Paste the DetectNet model description in the text area.
- In `Pretrained model(s)` specify the path to the pre-trained GoogLeNet.

You may click `Visualize` to review the network topology:

![click](click-visualize.jpg)

![detectnet](detectnet.jpg)

> NOTE: this instance of DetectNet requires at least 12GB of GPU memory.
If you have less memory on your GPU[s], you may want to decrease the batch size.
On a 4GB card, you can set the batch size to 2 and the batch accumulation to 5, for an effective batch of 10, and that should fit on your card.

![batch-accumulation](batch-accumulation.jpg)

Finally, select the number of GPUs to train on, give your model a name then click `Create`:

![select GPUs](select-gpus.jpg)

After training the model for 30 epochs the training curves may look like below.
Make good note of the purple curve which is showing the `mAP` (mean Average Precision).
The `mAP` is the main indicator of the network accuracy:

![training loss](training-loss.jpg)

### Verification

To assess the model accuracy we can verify how the model performs on test images.
The network output is better visualized by drawing bounding rectangles around detected objects.
To this avail, select `Bounding Boxes` in `Select Visualization Method`:

![select-visualization](select-visualization.jpg)

To test an image, in `Test a single Image`, specify the path to an image then click `Test One`.
The output may be rendered as below:

![test one](test-one.jpg)

You may also test multiple images at once by specifying the image paths in a text file (one line per image path).
To that end, in `Test a list of Images`, upload an image list.
The output may be rendered as below:

![test meny](test-many.jpg)

The options cog menu allows you to adjust a few view options.
The opacity applies to the interior of the bounding box rectangle,
and desaturation appies to the image, which is useful when the image contains a lot of the bounding box color.

![test meny](display-options-menu.jpg)
