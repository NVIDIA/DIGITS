# Training an image regression model using DIGITS

Table of Contents
=================
* [Introduction](#introduction)
* [Installing Image Gradients extensions](#installing-image-gradients-extensions)
* [Dataset creation](#dataset-creation)
    * [Using the Gradient Data Extension](#using-the-gradient-data-extension)
    * [Alternative method: manually creating LMDB files](#alternative-method-manually-creating-lmdb-files)
* [Model creation](#model-creation)
    * [Using Caffe](#using-caffe)
    * [Using Torch7](#using-torch7)
* [Verification](#verification)
    * [Visualizing the network output](#visualizing-the-network-output)

## Introduction

Image classification models aim to learn to predict the class of an image, where each class is a discrete element from a finite set.
Image regression models may learn to predict any number of image characteristics. These characteristics are typically represented as a matrix or a vector of real numbers.
DIGITS may be used to train image regression models. This page will walk you through a simple example where a model is trained to predict the `x` and `y` gradients of a linear image
(a linear image is an image that has constant gradients in the `x` and `y` directions - `x` and `y` gradients may be different though).

## Installing Image Gradients extensions

Extensions are thin interfaces thay may be used in DIGITS to implement custom methods for ingesting data and visualizing network outputs during inference.
DIGITS supports a number of built-in extensions.
Additionally, custom extensions can be wrapped in a plug-in and installed separately.
To install the image gradients plug-ins, you may proceed as follows:

### Installing DIGITS package

If you haven't done so already, install the main DIGITS package.
This only needs to be done once:

```sh
$ pip install -e $DIGITS_ROOT
```

### Installing data and view plug-ins for Image Gradients:

```
$ pip install $DIGITS_ROOT/plugins/data/imageGradients
$ pip install $DIGITS_ROOT/plugins/view/imageGradients
```

## Dataset Creation

### Using the Gradient Data Extension

Select the `Datasets` tab then click `New Dataset>Images>Gradients`:

![test image](select-gradient-data-extension.png)

On the dataset creation page, default values are suitable to follow this example though you may elect to change any of these.
In particular you may request larger images (e.g. `128x128`) to see the gradient more clearly during visualization.
When you are ready, give the dataset a name then click `Create`:

![create dataset using extension](create-dataset-using-extension.png)

### Alternative method: manually creating LMDB files

Non-classification datasets may be created in DIGITS through the "other" type of datasets. For these datasets, DIGITS expects the user to provide a set of LMDB databases.
Note that since labels may be vectors (or matrices), it is not possible to use a single LMDB database to hold the image and its label. Therefore DIGITS expects one LMDB database for the images and a separate LMDB database for the labels.

The first step in creating the dataset is to create the LMDB databases. In this example you will use the Python test script located in `/digits/dataset/images/generic/test_lmdb_creator.py`.
This script creates a number of grayscale linear images and adds them to a train database and a validation database. For each image, the `x` and `y` (normalized) gradients are chosen randomly from a uniform distribution `[-0.5,0.5)`.

To create a train database of 100 50x50 images:
```sh
$ ./digits/dataset/images/generic/test_lmdb_creator.py -x 50 -y 50 -c 100 /tmp/my_dataset
Creating images at "/tmp/my_dataset" ...
Done after 11.3920481205 seconds
```

The script also creates a validation database of 20 samples. Overall, the script creates train image and label databases, validation image and label
databases, train and validation mean images, and a test image.

See for example the `test.png` image which is created using gradients of 0.5 in both directions:

![test image](test.png)

Now that we have created the required files, we may create the dataset using DIGITS.
On the main page, select the `Datasets` tab then click `New Dataset>Images>Other`:

![Create generic dataset](create-generic-dataset.png)

In the generic dataset creation form you need to provide the paths to:
- the train image database
- the train label database
- the validation image database
- the validation label database
- the train mean image `train_mean.binaryproto` file

![Create generic dataset form](create-regression-dataset.png)

## Model creation

Now that you have a regression dataset to train on, you will create a regression model.
On the home page, select the `Models` tab then click `New Model>Images>Gradients` or `New Model>Images>Other`, depending on how you created the dataset.

![Create generic model](create-model.png)

On the model creation form, select the dataset you just created. We will be creating a very simple fully linear model that consists of
just one fully connected layer. You may use either Caffe or Torch7 to define the model.

### Using Caffe

Under the `Custom Network` tab, select `Caffe`. There you can paste the following network definition:
```protobuf
layer {
  name: "scale"
  type: "Power"
  bottom: "data"
  top: "scale"
  power_param {
    scale: 0.004
  }
}
layer {
  name: "hidden"
  type: "InnerProduct"
  bottom: "scale"
  top: "output"
  inner_product_param {
    num_output: 2
  }
}
layer {
  name: "loss"
  type: "EuclideanLoss"
  bottom: "output"
  bottom: "label"
  top: "loss"
  exclude { stage: "deploy" }
}
```

You may lower the base learning rate to `0.001` to ensure a smoother learning curve.

### Using Torch7
Under the `Custom Network` tab, select `Torch`. There you can paste the following network definition:
```lua
return function(p)
    local nDim=1
    if p.inputShape then p.inputShape:apply(function(x) nDim=nDim*x end) end
    local net = nn.Sequential()
    net:add(nn.MulConstant(0.004))
    net:add(nn.View(-1):setNumInputDims(3))
    net:add(nn.Linear(nDim,2))
    return {
        model = net,
        loss = nn.MSECriterion(),
    }
end
```

## Verification

After training for 15 epochs the loss function should look similar to this:

![Training loss](regression-loss.png))

Now we can assess the quality of the model. To this avail, we can use the test image that was generated by `test_lmdb_creator.py`:

![Test single image](regression-test-one.png)

A new window will appear showing the test image and the output of the network, which is `[ 0.50986129 0.48490545]` and close enough
to the real gradients used to create the test image (`[0.5, 0.5]`).

![Original image](regression-output.png)

### Visualizing the network output

The Gradient View Extension may be used to visualize the network output.
To this avail, in the `Select Visualization Method` section, select the `Gradients` extension:

![Select gradient extension](select-gradient-view-extension.png)

Use the validation database to test a list of images.
The validation database may be found within the `val_db/features` sub-folder of the dataset job folder:

![Test DB](test-db.png)

Click `Test DB`.
The output may be rendered as below.
You will notice that the arrow is rightly pointing in the direction of the gradient (i.e. towards the light) on those images:

![Test DB Inference](test-db-inference.png)
