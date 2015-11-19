#!/usr/bin/env python
# -*- coding: utf-8 -*-
import argparse
import pickle, skimage
import logging
import numpy as np
import os
import sys
import time

try:
    import digits
except ImportError:
    # Add path for DIGITS package
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import digits.config
digits.config.load_config()
from digits import log
# must call digits.config.load_config() before caffe to set the path
try:
    import caffe_pb2
except ImportError:
    # See issue #32
    from caffe.proto import caffe_pb2

from digits.config import config_value
import caffe
import lmdb
from StringIO import StringIO
import PIL

logger = logging.getLogger('digits.tools.compute_accuracy')

class DbReader(object):
    """
    Reads a database
    """

    def __init__(self, location):
        """
        Arguments:
        location -- where is the database
        """
        self._db = lmdb.open(location,
                map_size=1024**3, # 1MB
                max_dbs=0,
                readonly=True)

        with self._db.begin() as txn:
            self.total_entries = txn.stat()['entries']

    def entries(self):
        """
        Generator returning all entries in the DB
        """
        with self._db.begin() as txn:
            cursor = txn.cursor()
            for item in cursor:
                yield item



def validate_database_path(database):
    """
    Returns a valid database path
    Throws ValueErrors
    """
    p = os.path.abspath(database)
    if not os.path.exists(p):
        raise ValueError('No such file or directory')
    if os.path.isfile(p):
        p = os.path.dirname(p)
    if not os.path.isdir(p):
        raise ValueError('Not a directory')
    return p



class Classifier(caffe.Net):
    """
    Classifier extends Net for image class prediction
    by scaling, center cropping, or oversampling.
    It differs from Caffe original class version by
    taking the last output layer instead of the first (which
    is not always the right output layer, e.g in the case
    of googlenet)
    """
    def __init__(self, model_file, pretrained_file, image_dims=None,
                 mean=None, input_scale=None, raw_scale=None,
                 channel_swap=None):
        """
        Take
        image_dims: dimensions to scale input for cropping/sampling.
            Default is to scale to net input size for whole-image crop.
            mean, input_scale, raw_scale, channel_swap: params for
            preprocessing options.
        """
        caffe.Net.__init__(self, model_file, pretrained_file, caffe.TEST)

        # configure pre-processing
        in_ = self.inputs[0]
        self.transformer = caffe.io.Transformer(
            {in_: self.blobs[in_].data.shape})
        self.transformer.set_transpose(in_, (2,0,1))
        if mean is not None:
            self.transformer.set_mean(in_, mean)
        if input_scale is not None:
            self.transformer.set_input_scale(in_, input_scale)
        if raw_scale is not None:
            self.transformer.set_raw_scale(in_, raw_scale)
        if channel_swap is not None:
            self.transformer.set_channel_swap(in_, channel_swap)

        self.crop_dims = np.array(self.blobs[in_].data.shape[2:])
        if not image_dims:
            image_dims = self.crop_dims
        self.image_dims = image_dims


    def predict(self, inputs, out_layer_name=None, oversample=True):
        """
        Predict classification probabilities of inputs.

        Take
        inputs: iterable of (H x W x K) input ndarrays.
        oversample: average predictions across center, corners, and mirrors
                    when True (default). Center-only prediction when False.

        Give
        predictions: (N x C) ndarray of class probabilities
                     for N images and C classes.
        """
        # Scale to standardize input dimensions.
        input_ = np.zeros((len(inputs),
            self.image_dims[0], self.image_dims[1], inputs[0].shape[2]),
            dtype=np.float32)
        for ix, in_ in enumerate(inputs):
            input_[ix] = caffe.io.resize_image(in_, self.image_dims)

        if oversample:
            # Generate center, corner, and mirrored crops.
            input_ = caffe.io.oversample(input_, self.crop_dims)
        else:
            # Take center crop.
            center = np.array(self.image_dims) / 2.0
            crop = np.tile(center, (1, 2))[0] + np.concatenate([
                -self.crop_dims / 2.0,
                self.crop_dims / 2.0
            ])
            input_ = input_[:, crop[0]:crop[2], crop[1]:crop[3], :]

        # Classify
        caffe_in = np.zeros(np.array(input_.shape)[[0,3,1,2]],
                            dtype=np.float32)
        for ix, in_ in enumerate(input_):
            caffe_in[ix] = self.transformer.preprocess(self.inputs[0], in_)
        out = self.forward_all(**{self.inputs[0]: caffe_in})

        # Where it differs from the original Caffe class
        if out_layer_name == None:
            predictions = out[self.outputs[-1]]
        else:
            predictions = out[out_layer_name]

        # For oversampling, average predictions across crops.
        if oversample:
            predictions = predictions.reshape((len(predictions) / 10, 10, -1))
            predictions = predictions.mean(1)

        return predictions

def compute_accuracy(database, snapshot, deploy_file, mean_file, grayscale=False, oversample=False):
    """
    Evaluate a Net on a set of images, and dump the result in two
    pickle files.
    Returns True on sucess

    Arguments:
    snapshot -- a Caffe trained model
    deploy_file -- the corresponding deploy file
    labels_file -- the file containing the dataset labels
    mean_file -- the dataset mean file
    img_set -- the file containing a list of images
    resize_mode -- the mode used to resize images

    Keyword arguments:
    oversampling -- boolean, True if oversampling should be enabled
    """

    snapshot_file, _ = os.path.splitext(snapshot)
    if os.path.isfile(snapshot_file + "-accuracy-proba.pkl") and os.path.isfile(snapshot_file + "-accuracy-labels.pkl"):
        logger.debug("Done")
        return True


    # Open database
    try:
        database = validate_database_path(database)
    except ValueError as e:
        logger.error(e.message)
        return False

    reader = DbReader(database)


    # Loading the classifier
    if config_value('caffe_root')['cuda_enabled'] and config_value('gpu_list'):
        caffe.set_mode_gpu()

    mean_blob = caffe.proto.caffe_pb2.BlobProto()
    mean_blob.ParseFromString(open(mean_file, 'rb').read())
    mean = np.array(caffe.io.blobproto_to_array(mean_blob))[0].mean(1).mean(1)

    if grayscale:
        net = Classifier(deploy_file, snapshot,
                       mean=mean,
                       raw_scale=255)
    else:
        net = Classifier(deploy_file, snapshot,
                       mean=mean,
                       channel_swap=(2,1,0),
                       raw_scale=255)


    size = reader.total_entries

    labels = []
    probas = []

    counter = 0

    for key, value in reader.entries():
        datum = caffe_pb2.Datum()
        datum.ParseFromString(value)

        s = StringIO()
        s.write(datum.data)
        s.seek(0)
        input_image = PIL.Image.open(s)

        if grayscale:
            input_image = input_image.astype(np.uint8)
            input_image = input_image.reshape(input_image.shape[0], input_image.shape[1], 1)
        else:
            if input_image.mode == 'L':
                input_image = input_image.convert('RGB')
            input_image = skimage.img_as_float(input_image).astype(np.float32)

        probas.append(net.predict([input_image], oversample=False)[0])
        labels.append(datum.label)
        counter += 1

        if size > 500 and counter % (size/500) == 0:
            logger.debug("Progress: %0.2f" % (counter/float(size)))


    # Dumping the result
    pickle.dump(probas, open(snapshot_file + "-accuracy-proba.pkl", "wb"))
    pickle.dump(labels, open(snapshot_file + "-accuracy-labels.pkl", "wb"))
    logger.debug("Done")
    return True


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Accuracy-Computing tool - DIGITS')
    parser.add_argument('database',
            help='The lmdb database.'
            )
    parser.add_argument('snapshot',
            help='The Caffe model snapshot.'
            )

    parser.add_argument('deploy_file',
            help='The deploy_file.'
            )

    parser.add_argument('mean_file',
            help='The dataset mean file.'
            )

    parser.add_argument("--grayscale",
            help="grayscale mode", action="store_true")

    args = vars(parser.parse_args())
    start_time = time.time()

    if compute_accuracy(
        args['database'],
        args['snapshot'],
        args['deploy_file'],
        args['mean_file'],
        grayscale=args['grayscale']):
        logger.info('Done after %d seconds.' % (time.time() - start_time))
        sys.exit(0)
    else:
        sys.exit(1)
