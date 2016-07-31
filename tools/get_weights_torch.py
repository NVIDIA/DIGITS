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
import digits

from digits.config import load_config
load_config()

from digits import utils, log
from digits.inference.errors import InferenceError
from digits.framework_helpers import torch_helpers


logger = logging.getLogger('digits.tools.inference')

"""
Perform inference on a list of images using the specified model
"""

def get_weights(output_dir):

    f = h5py.File(output_dir+'/filters.hdf5','a')
    vis_file = h5py.File(output_dir+'/weights.h5','r')

    # Save param keys to file:
    num_outputs = len(vis_file['layers'].keys())
    for index, key in enumerate(vis_file['layers'].keys()):
        if 'weights' in vis_file['layers'][key]:
            chain = key
            raw_data = vis_file['layers'][chain]['weights'][...]

            if len(raw_data.shape)>1 and raw_data.shape[0]==1:
                raw_data = raw_data[0]

            vis_data = utils.image.reshape_data_for_vis(raw_data,'BGR')
            dset  = f.create_dataset(chain, data=utils.image.normalize_data(vis_data))
            dset.attrs['stats'] = json.dumps({"shape": raw_data.shape})

        logger.info('Processed %s/%s layers', index, num_outputs)

    vis_file.close()
    f.close()

def infer(model_def_path, weights_path, gpu):

    torch_helpers.save_weights(
            model_def_path,
            weights_path,
            gpu,
            logger
            )

    get_weights(os.path.split(model_def_path)[0])
    logger.info('Saved data to %s', os.path.split(model_def_path)[0])

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
            args['model_def_path'],
            args['weights_path'],
            args['gpu']
            )
    except Exception as e:
        logger.error('%s: %s' % (type(e).__name__, e.message))
        raise
