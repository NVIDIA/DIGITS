#!/usr/bin/env python2
# Copyright (c) 2016, NVIDIA CORPORATION.  All rights reserved.

import argparse
import h5py
import logging
import numpy as np
import os
import sys
import cv2
from scipy.optimize import curve_fit
from scipy import exp

# Add path for DIGITS package
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import digits.config
digits.config.load_config()

from digits import utils, log
from digits.inference.errors import InferenceError
from digits.framework_helpers import torch_helpers

logger = logging.getLogger('digits.tools.inference')

"""
Get max activations from a pretrained model
"""
def run(model_def_path, weights_path, height, width, layer,units=[0], mean_file_path=None,gpu=None):

    torch_helpers.save_max_activations(
        model_def_path,
        weights_path,
        height,width,
        layer,
        units,
        mean_file_path,
        gpu,
        logger
        )

    logger.info('Saved data to %s', os.path.split(model_def_path)[0])

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Gradient Ascent tool for pretrained models - DIGITS')

    ### Positional arguments
    parser.add_argument('-o', '--output_dir',
            help='Output directory',
            default='.'
            )

    parser.add_argument('-p', '--model_def_path',
            help='Path to model definition.prototxt',
            )

    parser.add_argument('-w', '--weights_path',
            help='Path to weights.caffemodel',
            )

    parser.add_argument('-x', '--height',
            help='Height of Input',
            )
    parser.add_argument('-y', '--width',
            help='Width of Input',
            )
    parser.add_argument('-l', '--layer',
            help='Name of output layer',
            )
    parser.add_argument('-m', '--mean_file_path',
            help='Mean file',
            )
    parser.add_argument('-u', '--units',
            type=str,
            default="-1",
            help='Index of units to optimize in output layer',
            )

    ### Optional arguments
    parser.add_argument('-g', '--gpu',
            type=int,
            default=None,
            help='GPU to use (as in nvidia-smi output, default: None)',
            )

    args = vars(parser.parse_args())

    try:
        run(
            args['model_def_path'],
            args['weights_path'],
            args['height'],
            args['width'],
            args['layer'],
            map(int,str.split(args['units'],",")),
            args['mean_file_path'],
            args['gpu']
            )
    except Exception as e:
        logger.error('%s: %s' % (type(e).__name__, e.message))
        raise
