#!/usr/bin/env python2
# Copyright (c) 2016, NVIDIA CORPORATION.  All rights reserved.

import argparse
import h5py
import json
import logging
import numpy as np
import os
import sys

# Add path for DIGITS package
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import digits.config
digits.config.load_config()

from digits import utils, log
from digits.inference.errors import InferenceError

# must call digits.config.load_config() before caffe to set the path
import caffe

logger = logging.getLogger('digits.tools.inference')

"""
Get weights from a pretrained model
"""

def get_weights(output_dir,net):

    f = h5py.File(output_dir+'/filters.hdf5','a')

    # load the image and perform a forward pass:
    try:
        net.forward()

        # Save param keys to file:
        num_outputs = len(net.params.keys())
        for index, key in enumerate(net.params.keys()):
            raw_data = net.params[key][0].data

            vis_data = utils.image.reshape_data_for_vis(raw_data,'BGR')
            dset = f.create_dataset(key, data=utils.image.normalize_data(vis_data))
            # TODO: Add more stats
            dset.attrs['stats'] = json.dumps({"shape": raw_data.shape})

            logger.info('Processed %s/%s blobs', index, num_outputs)

    except utils.errors.LoadImageError as e:
        print e

    f.close()

def infer(output_dir, model_def_path, weights_path, gpu):

    if gpu is not None:
        caffe.set_device(gpu)
        caffe.set_mode_gpu()
    else:
        caffe.set_mode_cpu()

    net = caffe.Net(model_def_path,weights_path,caffe.TEST)

    get_weights(output_dir, net)

    logger.info('Saved data to %s', output_dir)

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Weights tool for pretrained models - DIGITS')

    ### Positional arguments
    parser.add_argument('output_dir',
            help='Directory to write outputs to')

    parser.add_argument('-p', '--model_def_path',
            help='Path to model definition',
            )

    parser.add_argument('-m', '--weights_path',
            help='Path to weights',
            )

    ### Optional arguments
    parser.add_argument('-g', '--gpu',
            type=int,
            default=None,
            help='GPU to use (as in nvidia-smi output, default: None)',
            )

    args = vars(parser.parse_args())

    try:
        infer(
            args['output_dir'],
            args['model_def_path'],
            args['weights_path'],
            args['gpu']
            )
    except Exception as e:
        logger.error('%s: %s' % (type(e).__name__, e.message))
        raise
