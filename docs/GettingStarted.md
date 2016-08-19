# Getting Started

Now that you have successfully installed DIGITS, this guide will teach you the basics of how to use it.
By the end, you will have trained a Caffe model to recognize hand-written digits.
We will be using the [MNIST handwritten digit database](http://yann.lecun.com/exdb/mnist) as our dataset and [LeNet-5](http://yann.lecun.com/exdb/lenet/) for our network.
Both are generously made available by Yann LeCun on [his website](http://yann.lecun.com/).

## Download the data

Use the following command to download the MNIST dataset onto your server
(for Deb package installations, the script is at `/usr/share/digits/tools/download_data/main.py`):
```sh
$ $DIGITS_HOME/tools/download_data/main.py mnist ~/mnist
Downloading url=http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz ...
Downloading url=http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz ...
Downloading url=http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz ...
Downloading url=http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz ...
Uncompressing file=train-images-idx3-ubyte.gz ...
Uncompressing file=train-labels-idx1-ubyte.gz ...
Uncompressing file=t10k-images-idx3-ubyte.gz ...
Uncompressing file=t10k-labels-idx1-ubyte.gz ...
Reading labels from /home/username/mnist/train-labels.bin ...
Reading images from /home/username/mnist/train-images.bin ...
Reading labels from /home/username/mnist/test-labels.bin ...
Reading images from /home/username/mnist/test-images.bin ...
Dataset directory is created successfully at '/home/username/mnist'
Done after 16.722807169 seconds.
```

See [Standard Datasets](StandardDatasets.md) for details about this script.

## Using the Webapp

Open up a web browser and navigate to the home screen of DIGITS.
The server should be at either `http://localhost/` (if installed from Deb packages), `http://localhost:5000/` (if using `digits-devserver`) or `http://localhost:34448/` (if using `digits-server`).

![Home page](images/home-page-1.jpg)

### Logging in

Click on `New Dataset > Images > Classification`.
This will lead you to the login page:

> NOTE: there is no authentication - you don't even need a password.
This is a utility feature, not a security feature.

![Login](images/login.jpg)

### Creating a Dataset

After logging in, you will be brought to the "New Image Classification Dataset" page.

* Type in the path to the MNIST training images
  * You can also add the folder of MNIST test images as a "Separate validation images folder", if you like. Don't use the "test images" fields - test images are not used for anything in DIGITS yet.
* Change the `Image Type` to `Grayscale`
* Change the `Image size` to 28 x 28
* Give the dataset a name
* Click on the `Create` button

![New dataset](images/new-dataset.jpg)

While the job is running, you should see the expected completion time on the right side:

![Creating dataset](images/creating-dataset.jpg)

When the job is finished, go back to the home page by clicking `DIGITS` in the top left hand part of the page.
Then click on the "Datasets" tab.
You should now see your dataset listed.

![Home page with dataset](images/home-page-2.jpg)

### Training a Model

Click on `New Model > Images > Classification`.
This will lead you to the "New Image Classification Model" page.

For this example, do the following:
* Choose the "MNIST" dataset in the `Select Dataset` field
* Choose the `LeNet` network in the `Standard Networks` tab
* Give the model a name
* Click on the `Create` button

![New model (top)](images/new-model-top-half.jpg)
![New model (bottom)](images/new-model-bottom-half.jpg)

While training the model, you should see the expected completion time on the right side:

![Training model](images/training-model.jpg)

To test the model, scroll to the bottom of the page.
* Click on the `Upload image` button and choose a file
  * There are plenty to choose from in `/home/username/mnist/test/`
* Or, find an image on the web and paste the URL into the `Image URL` field
* Check the `Show visualizations and statistics` box
* Click on `Classify One`

![Classifying one image](images/classifying-one-image.jpg)

At the top of the page, DIGITS displays the top five classifications and corresponding confidence values.
DIGITS also provides visualizations and statistics about the weights and activations of each layer in your network.

![Classified one image](images/classified-one-image.jpg)

## More Guides

Once you have finished this guide, take a look at some of the other documentation at [docs/](.) and [examples/](../examples/):

* [Getting Started with Torch7](GettingStartedTorch.md)
* [Fine-tune a pretrained model](../examples/fine-tuning/README.md)
* [Train an autoencoder network](../examples/autoencoder/README.md)
* [Train a regression network](../examples/regression/README.md)
* [Train a Siamese network](../examples/siamese/README.md)
* [Train a text classification network](../examples/text-classification/README.md)
* [Learn more about weight initialization](../examples/weight-init/README.md)
* [Use Python layers in your Caffe networks](../examples/python-layer/README.md)
* [Download a model and use it to classify an image outside of DIGITS](../examples/classification/README.md)
* [Overview of the REST API](API.md)
