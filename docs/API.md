# Using the DIGITS REST API

Table of Contents
=================
* [Introduction](#introduction)
* [Image classification](#image-classification)
    * [Creating the classification dataset](#creating-the-classification-dataset)
    * [Creating the classification model](#creating-the-classification-model)
    * [Classification](#classification)
    * [Deleting a job](#deleting-a-job)
* [Regression](#regression)
    * [Creating the regression dataset](#creating-the-regression-dataset)
    * [Creating the regression model](#creating-the-regression-model)
    * [Inference](#inference)

## Introduction

The DIGITS REST API is a programming interface to DIGITS.
In the spirit of [REST](https://en.wikipedia.org/wiki/Representational_state_transfer), the API may be used to make self-contained, stateless queries to DIGITS.
In particular, the API may be used to create datasets and models, retrieve job information and perform inference on a trained model.
Besides, this interface is easily scriptable, which allows for actions to be performed using DIGITS programmatically.

In the first part of this walk-through we will see how the API may be used to create an image classification model.
In the second part we will see how to create a regression model.

> NOTE: this is not a comprehensive guide to the DIGITS REST API.
> You should be able to refer to the code if in doubt about a specified feature.

We will be using the `curl` command to interact with DIGITS from command line.
You may use any URL access library to carry out the same actions from your favorite programming language.

We will be assuming DIGITS is running on `localhost`.
If DIGITS is running on another host/port, please adjust the command lines accordingly.

## Image classification

This is another way of running the [Getting started](GettingStarted.md) tutorial.
If you haven't gone through this tutorial, please do so now.

### Creating the classification dataset

In order to create a dataset, you will first need to log in.
The following command will log us in as user `fred`:

```sh
$ curl localhost/login -c digits.cookie -XPOST -F username=fred
<!DOCTYPE HTML PUBLIC "-//W3C//DTD HTML 3.2 Final//EN">
<title>Redirecting...</title>
<h1>Redirecting...</h1>
<p>You should be redirected automatically to target URL: <a href="/">/</a>.  If not click the link.(venv)
```

Note the `-c digits.cookie` flag, which instructs `curl` to store the session cookie into `digits.cookie`.
DIGITS requires users to log in before creating jobs.
A job can only be edited or deleted by the user that created it.
The session cookie is required for all commands that create or modify jobs.
For those commands we will use `-b digits.cookie` in the `curl` command line to pass the session cookie to DIGITS.

> NOTE: if you prefer not to store cookies you may to use the `username` hidden form field directly.
> This may be done by replacing `-b digits.cookie` with `-F username=fred` in the commands that require authentication.
> Using cookies would however be more robust to future changes in the authentication scheme.

In the above command `/login` is referred to as a "route".
Every route in DIGITS has a different function.
Refer to `/digits/webapp.py` for a list of route entry points.

Assuming you have already downloaded the MNIST dataset as per the Getting Started tutorial, you can create a dataset named `mnist_dataset` by running this command:

```sh
$ export MNIST_PATH=/path/to/mnist
$ curl localhost/datasets/images/classification.json -b digits.cookie -XPOST -F folder_train=$MNIST_PATH/train -F encoding=png -F resize_channels=1 -F resize_width=28 -F resize_height=28 -F method=folder -F dataset_name=mnist_dataset
{
  "id": "20160809-103210-ccbf",
  "name": "mnist_dataset",
  "status": "Initialized"
}
```

Here you can see how we set the value of various fields in the dataset creation form.
All form fields can be specified using the REST API.
For a comprehensive list of the available form fields, refer to the code on `digits/dataset/images/classification/forms.py` and the corresponding parent classes.

In the response that we received from DIGITS, the job ID for this classification dataset is printed as `20160809-103210-ccbf`.
Now we can check the status of this job by doing:

```sh
$ curl localhost/datasets/20160809-103210-ccbf/status
{"status": "Done", "type": "Image Classification Dataset", "name": "mnist_dataset", "error": null}
```

The above output indicates that the job has completed.
We can now get more detailed job information by doing:

```sh
$ curl localhost/datasets/20160809-103210-ccbf.json
{
  "CreateDbTasks": [
    {
      "backend": "lmdb",
      "compression": "none",
      "encoding": "png",
      "entries": 45002,
      "image_channels": 1,
      "image_height": 28,
      "image_width": 28,
      "name": "Create DB (train)"
    },
    {
      "backend": "lmdb",
      "compression": "none",
      "encoding": "png",
      "entries": 14998,
      "image_channels": 1,
      "image_height": 28,
      "image_width": 28,
      "name": "Create DB (val)"
    }
  ],
  "ParseFolderTasks": [
    {
      "label_count": 10,
      "name": "Parse Folder (train/val)",
      "test_count": 0,
      "train_count": 45002,
      "val_count": 14998
    }
  ],
  "directory": "/home/greg/ws/digits/digits/jobs/20160809-103210-ccbf",
  "id": "20160809-103210-ccbf",
  "name": "mnist_dataset",
  "status": "Done"
}
```

You can also use the `/index.json` route to list all existing jobs:

```sh
$ curl localhost/index.json
{
  "datasets": [
    {
      "id": "20160809-103957-6d37",
      "name": "mnist_dataset",
      "status": "Done"
    }
  ],
  "models": [
  ],
  "version": "4.1-dev"
}
```

### Creating the classification model

Now that we have a dataset we may create the model:

```sh
$  curl localhost/models/images/classification.json -b digits.cookie -XPOST -F method=standard -F standard_networks=lenet -F train_epochs=30 -F framework=caffe -F model_name=lnet_mnist -F dataset=20160809-103957-6d37
{
  "caffe flavor": "NVIDIA",
  "caffe version": "0.15.9",
  "creation time": "2016-08-09 11:24:14.853354",
  "dataset_id": "20160809-103957-6d37",
  "deploy file": "deploy.prototxt",
  "digits version": "4.1-dev",
  "framework": "caffe",
  "id": "20160809-112414-0296",
  "image dimensions": [
    28,
    28,
    1
  ],
  "image resize mode": "squash",
  "job id": "20160809-112414-0296",
  "labels file": "labels.txt",
  "mean file": "mean.binaryproto",
  "name": "lenet_mnist",
  "network file": "original.prototxt",
  "snapshot file": "no snapshots",
  "solver file": "solver.prototxt",
  "status": "Initialized",
  "train_val file": "train_val.prototxt",
  "username": "fred"
}
```

For more information on the parameters to the model creation form, refer to the code on `digits/model/images/classification/forms.py` and the corresponding parent classes.

While the model is being trained we can query its status:

```sh
$ curl localhost/models/20160809-112414-0296.json
{
  "caffe flavor": "NVIDIA",
  "caffe version": "0.15.9",
  "creation time": "2016-08-09 11:24:14.853354",
  "dataset_id": "20160809-103957-6d37",
  "deploy file": "deploy.prototxt",
  "digits version": "4.1-dev",
  "directory": "/home/greg/ws/digits/digits/jobs/20160809-112414-0296",
  "framework": "caffe",
  "id": "20160809-112414-0296",
  "image dimensions": [
    28,
    28,
    1
  ],
  "image resize mode": "squash",
  "job id": "20160809-112414-0296",
  "labels file": "labels.txt",
  "mean file": "mean.binaryproto",
  "name": "lenet_mnist",
  "network file": "original.prototxt",
  "snapshot file": "snapshot_iter_15488.caffemodel",
  "snapshots": [
    1,
    ...
    22
  ],
  "solver file": "solver.prototxt",
  "status": "Running",
  "train_val file": "train_val.prototxt",
  "username": "fred"
}
```

### Classification

Now that we have a trained model we can classify an image by using the `/models/images/classification/classify_one.json` route:

```sh
$ curl localhost/models/images/classification/classify_one.json -XPOST -F job_id=20160809-112414-0296 -F image_file=@$MNIST_PATH/test/0/00003.png
{
  "predictions": [
    [
      "0",
      100.0
    ],
    [
      "6",
      0.0
    ],
    [
      "2",
      0.0
    ],
    [
      "5",
      0.0
    ],
    [
      "8",
      0.0
    ]
  ]
}
```

We can also classify many images at once by using the `/models/images/classification/classify_many.json` route.
We can get the image list from the validation set in the dataset job folder:

```
$ curl localhost/models/images/classification/classify_many.json -XPOST -F job_id=20160809-112414-0296 -F image_list=@/home/greg/ws/digits/digits/jobs/20160809-103957-6d37/va.txt > predictions.txt
  % Total    % Received % Xferd  Average Speed   Time    Time     Time  Current
                                 Dload  Upload   Total   Spent    Left  Speed
100 4880k  100 4162k  100  717k   631k   108k  0:00:06  0:00:06 --:--:--  959k
```

All classifications will be stored in the file `predictions.txt`.

### Deleting a job

If you do not need the model anymore, you can delete it by using the `DELETE` action on the `/models/<job_id>` route:

```sh
$ curl localhost/models/20160809-112414-0296 -b digits.cookie -X DELETE
Job deleted.
```

## Regression

This is another way of running the [regression](../examples/regression/README.md) tutorial.
If you haven't gone through this tutorial, please do so now.

### Creating the regression dataset

We will use the gradients extension to easily create a dataset of gradient images.
The regression dataset may be created by using the `/datasets/generic/create/<extension_id>.json` route.
The `extension_id` for the gradients extension is `image-gradients`.
The following command creates a dataset named `gradient_dataset` with all default parameters:

```sh
$ curl localhost/datasets/generic/create/image-gradients.json -b digits.cookie -X POST -F dataset_name=gradient_dataset
{
  "id": "20160809-115121-bf13",
  "name": "gradient_dataset",
  "status": "Initialized"
}
```

For more information on the parameters to generic dataset creation form, refer to the code on `digits/dataset/generic/forms.py` and the corresponding parent classes.
For more information on the parameters to the gradients extension creation form, refer to the code on `forms.py`, within the gradient extension code.

We can query the status of the dataset creation job by using the same route as in the image classification case:

```sh
$ curl localhost/datasets/20160809-115121-bf13.json
{
  "create_db_tasks": [
    {
      "entry_count": 1000,
      "feature_db_path": "train_db/features",
      "label_db_path": "train_db/labels",
      "name": "Create train_db DB",
      "stage": "train_db"
    },
    {
      "entry_count": 250,
      "feature_db_path": "val_db/features",
      "label_db_path": "val_db/labels",
      "name": "Create val_db DB",
      "stage": "val_db"
    },
    {
      "entry_count": 0,
      "feature_db_path": null,
      "label_db_path": null,
      "name": "Create test_db DB",
      "stage": "test_db"
    }
  ],
  "directory": "/home/greg/ws/digits/digits/jobs/20160809-115121-bf13",
  "feature_dims": [
    32,
    32,
    1
  ],
  "id": "20160809-115121-bf13",
  "name": "gradient_dataset",
  "status": "Done"
}
```

### Creating the regression model

Now that we have a dataset, we can create the model by using the `/models/images/generic.json` route.
We will be using Torch in this example, as specified by `-F framework=torch`.
In order to use a custom network, write the model description into a file (in this example `model.lua`).
Then type the following command:

```sh
$ curl localhost/models/images/generic.json -b digits.cookie -X POST -F method=custom -F train_epochs=3 -F framework=torch -F model_name=gradients_model -F dataset=20160809-115121-bf13 -F custom_network="$(<model.lua)"
{
  "creation time": "2016-08-09 12:19:39.867502",
  "dataset_id": "20160809-115121-bf13",
  "digits version": "4.1-dev",
  "framework": "torch",
  "id": "20160809-121939-ab6f",
  "image dimensions": [
    32,
    32,
    1
  ],
  "job id": "20160809-121939-ab6f",
  "mean file": "mean.binaryproto",
  "model file": "model.lua",
  "name": "gradients_model",
  "snapshot file": "no snapshots",
  "status": "Initialized",
  "username": "fred"
}
```

For more information on the parameters to the generic image model creation form, refer to the code on `digits/model/images/generic/forms.py` and the corresponding parent classes.

### Inference

Now that we have a model we can extract features from a test image using the `/models/images/generic/infer_one.json` route:

```sh
$ curl localhost/models/images/generic/infer_one.json -XPOST -F job_id=20160809-121939-ab6f -F image_file=@/home/greg/ws/digits/examples/regression/test.png
{
  "outputs": {
    "output": [
      [
        0.50042861700058,
        0.50376439094543
      ]
    ]
  }
}
```

It is also possible to perform multiple image inference using the `/models/images/generic/infer_many.json` route.
This is left as an exercise to the reader.
