# Classification Example

This example shows how to use python to consume a DIGITS model and classify images.
You should see the exact same numerical answers between DIGITS and these scripts (but take note of the "Limitations" section).

## Limitations

There are a few "gotchas" which can lead to discrepancies between DIGITS classifications and the output from this example.

##### Image resizing
This example resizes images using the `Squash` method.
In DIGITS, the same method will be used as was originally used when creating your dataset (`Crop`, `Squash`, `Fill` or `HalfCrop`).

##### Mean subtraction
This example subtracts a mean pixel rather than the whole mean file.
In DIGITS, the same method will be used as was originally used when training your model (`None`, `Image` or `Pixel`).

## Requirements

See [BuildCaffe.md](../../docs/BuildCaffe.md) for instructions about installing caffe.
Other requirements can be found in `requirements.txt`.
You do not need as many packages to run this example as you do to run DIGITS.

## Usage

Use one of the two python scripts provided to classify an image.

### Using a model archive

Use `use_archive.py` to classify images using a model archive downloaded from DIGITS (e.g. `20150512-171624-d9a9_epoch_30.tar.gz`).

```
$ ./use_archive.py -h
usage: use_archive.py [-h] [--nogpu] archive image

Classification example using an archive - DIGITS

positional arguments:
  archive     Path to a DIGITS model archive
  image       Path to an image

optional arguments:
  -h, --help  show this help message and exit
  --nogpu     Don't use the GPU

$ ./use_archive.py digits-model.tar.gz test-image.jpg
Extracting tarfile ...
Processed 1/1 images ...
Classification took 0.00310683250427 seconds.
--------------------------- Prediction for image.jpg ---------------------------
 96.1199% - "0"
  1.3588% - "6"
  0.7247% - "9"
  0.4695% - "2"
  0.3857% - "3"

Script took 0.270452022552 seconds.
```

### Using individual model files

If you have already extracted your model, you can specify each of the files manually with `example.py`.

```
$ ./example.py -h
usage: example.py [-h] [-m MEAN] [-l LABELS] [--nogpu]
               caffemodel deploy_file image

Classification example - DIGITS

positional arguments:
  caffemodel            Path to a .caffemodel
  deploy_file           Path to the deploy file
  image                 Path to an image

optional arguments:
  -h, --help            show this help message and exit
  -m MEAN, --mean MEAN  Path to a mean file (*.npy)
  -l LABELS, --labels LABELS
                        Path to a labels file
  --nogpu               Don't use the GPU

$ ./example.py snapshot_iter_1000.caffemodel deploy.prototxt test-image.jpg --mean mean.binaryproto --labels labels.txt
Processed 1/1 images ...
Classification took 0.00309991836548 seconds.
--------------------------- Prediction for image.jpg ---------------------------
 96.1199% - "0"
  1.3588% - "6"
  0.7247% - "9"
  0.4695% - "2"
  0.3857% - "3"

Script took 0.269672870636 seconds.
```

## Extensions

This example is kept pretty basic.
The user is encouraged to extend it to suit their own purposes.

#### Multiple images

The code already supports classifying a list of images, you just have to provide your own code to give the list of filenames.

#### Batched inference

Already done, just provide a value other than 1 for `batch_size` in `classify()`.

#### Filter visualization

See Caffe's example for extracting filters from a trained model: https://github.com/BVLC/caffe/blob/rc2/examples/filter_visualization.ipynb

