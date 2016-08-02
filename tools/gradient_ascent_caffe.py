#!/usr/bin/env python2
# Copyright (c) 2016, NVIDIA CORPORATION.  All rights reserved.

import argparse
import h5py
import json
import logging
import numpy as np
import os
import sys
import cv2

# Add path for DIGITS package
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import digits.config
digits.config.load_config()

from digits import utils, log
from digits.inference.errors import InferenceError

from tools.gradient_ascent.gradient_optimizer import GradientOptimizer, FindParams
import tools.gradient_ascent.caffe_misc as caffe_misc
# must call digits.config.load_config() before caffe to set the path
import caffe

logger = logging.getLogger('digits.tools.inference')

"""
Get max activations from a pretrained model
"""
def get_mean(mean_file_path, data_size):
    # Get mean from binaryprotot
    mean_blob = caffe.proto.caffe_pb2.BlobProto()
    mean_proto = open( mean_file_path , 'rb' ).read()
    mean_blob.ParseFromString(mean_proto)

    # Convert Mean Blob for Resizing with cv2 to input dimensions
    mean_image = np.transpose(np.array( caffe.io.blobproto_to_array(mean_blob) )[0], (1,2,0))
    data_mean = np.transpose(cv2.resize(mean_image,data_size), (2,0,1))

    return data_mean

def infer(model_def_path, weights_path, layer,unit, mean_file_path=None, gpu=None):

    if gpu is not None:
        caffe.set_device(gpu)
        caffe.set_mode_gpu()
    else:
        caffe.set_mode_cpu()

    # TODO : Make channel swap inputable
    net = caffe.Classifier(
        model_def_path,
        weights_path,
        channel_swap = (2,1,0)
    )

    # Get the input shape in order to resize mean image:
    in_ = net.inputs[0]
    input_shape = net.blobs[in_].data.shape
    data_size = (input_shape[2], input_shape[3])

    if mean_file_path is not None:
        # If mean entered, use is:
        mean = get_mean(mean_file_path, data_size)
    else:
        # Else generate grey image:
        mean = np.ones(net.blobs[in_].data[0].shape) * 150

    # Set the mean for the network (as it wasnt set during initialization)
    transformer = caffe.io.Transformer({in_: input_shape})
    transformer.set_mean(in_, mean)

    out = net.blobs[layer].data

    # Check if fully convolutional layer, or a convolutional layer
    # If convolutional set spacial to be the center to avoid cropping
    is_conv = (len(out.shape) == 4)
    if (is_conv):
        push_spatial = (out.shape[2]/2, out.shape[3]/2)
    else:
        push_spatial = (0,0)

    optimizer = GradientOptimizer(net,mean,channel_swap_to_rgb = (2,1,0))

    # TODO: Make the params below optionable:
    params = FindParams(
        push_layer = layer,
        push_channel = unit,
        decay = 0.001,
        blur_radius = 1.0,
        blur_every = 4,
        max_iter = 10,
        push_spatial = push_spatial,
        lr_params = {'lr': 100.0}
    )

    im = optimizer.run_optimize(params, prefix_template = "blah",brave = True,save=False)
    cv2.imshow('gradient',im)
    cv2.waitKey(0)

    logger.info('Saved data to %s', 'somewhere :/')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Weights tool for pretrained models - DIGITS')

    ### Positional arguments
    parser.add_argument('-p', '--model_def_path',
            help='Path to model definition.prototxt',
            )
    parser.add_argument('-w', '--weights_path',
            help='Path to weights.caffemodel',
            )

    parser.add_argument('-l', '--layer',
            help='Name of output layer',
            )
    parser.add_argument('-u', '--unit',
            type=int,
            help='Name of unit to optimize in output layer',
            )

    ### Optional arguments
    parser.add_argument('-m', '--mean_file_path',
            default=None,
            help='Path to mean.binaryproto',
            )

    parser.add_argument('-g', '--gpu',
            type=int,
            default=None,
            help='GPU to use (as in nvidia-smi output, default: None)',
            )

    args = vars(parser.parse_args())

    try:
        infer(
            args['model_def_path'],
            args['weights_path'],
            args['layer'],
            args['unit'],
            args['mean_file_path'],
            args['gpu']
            )
    except Exception as e:
        logger.error('%s: %s' % (type(e).__name__, e.message))
        raise
