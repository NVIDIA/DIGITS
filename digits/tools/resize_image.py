#!/usr/bin/env python2
# Copyright (c) 2014-2017, NVIDIA CORPORATION.  All rights reserved.

import argparse
import logging
import os
import sys

import PIL.Image

# Add path for DIGITS package
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import digits.config  # noqa
from digits import utils, log  # noqa

logger = logging.getLogger('digits.tools.resize_image')


def validate_output_file(filename):
    if filename is None:
        return True
    if os.path.exists(filename):
        if not os.access(filename, os.W_OK):
            logger.error('cannot overwrite existing output file "%s"' % filename)
            return False
    output_dir = os.path.dirname(filename)
    if not output_dir:
        output_dir = '.'
    if not os.path.exists(output_dir):
        logger.error('output directory "%s" does not exist' % output_dir)
        return False
    if not os.access(output_dir, os.W_OK):
        logger.error('you do not have write access to output directory "%s"' % output_dir)
        return False
    return True


def validate_input_file(filename):
    if not os.path.exists(filename) or not os.path.isfile(filename):
        logger.error('input file "%s" does not exist' % filename)
        return False
    if not os.access(filename, os.R_OK):
        logger.error('you do not have read access to "%s"' % filename)
        return False
    return True


def validate_range(number, min_value=None, max_value=None, allow_none=False):
    if number is None:
        if allow_none:
            return True
        else:
            logger.error('invalid value %s' % number)
            return False
    try:
        float(number)
    except ValueError:
        logger.error('invalid value %s' % number)
        return False

    if min_value is not None and number < min_value:
        logger.error('invalid value %s' % number)
        return False
    if max_value is not None and number > max_value:
        logger.error('invalid value %s' % number)
        return False
    return True

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Resize-Image tool - DIGITS')

    # Positional arguments

    parser.add_argument('image',
                        help='A filesystem path or url to the image'
                        )
    parser.add_argument('output',
                        help='The location to output the image'
                        )
    parser.add_argument('width',
                        type=int,
                        help='The new width'
                        )
    parser.add_argument('height',
                        type=int,
                        help='The new height'
                        )

    # Optional arguments

    parser.add_argument('-c', '--channels',
                        type=int,
                        help='The new number of channels [default is to remain unchanged]'
                        )
    parser.add_argument('-m', '--mode',
                        default='squash',
                        help='Resize mode (squash/crop/fill/half_crop) [default is squash]'
                        )

    args = vars(parser.parse_args())

    for valid in [
            validate_range(args['width'], min_value=1),
            validate_range(args['height'], min_value=1),
            validate_range(args['channels'],
                           min_value=1, max_value=3, allow_none=True),
            validate_output_file(args['output']),
    ]:
        if not valid:
            sys.exit(1)

    # load image
    image = utils.image.load_image(args['image'])

    # resize image
    image = utils.image.resize_image(image, args['height'], args['width'],
                                     channels=args['channels'],
                                     resize_mode=args['mode'],
                                     )
    image = PIL.Image.fromarray(image)
    try:
        image.save(args['output'])
    except KeyError:
        logger.error('Unable to save file to "%s"' % args['output'])
        sys.exit(1)
