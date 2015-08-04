#!/usr/bin/env python
"""
Functions for creating temporary datasets
Used in test_views
"""

import os
import time
import argparse
from collections import defaultdict

import numpy as np
import PIL.Image


IMAGE_SIZE  = 10
IMAGE_COUNT = 10 # per category


def create_classification_imageset(folder, image_size=None, image_count=None):
    """
    Creates a folder of folders of images for classification
    """
    if image_size is None:
        image_size = IMAGE_SIZE
    if image_count is None:
        image_count = IMAGE_COUNT

    # Stores the relative path of each image of the dataset
    paths = defaultdict(list)

    for class_name, pixel_index, rotation in [
            ('red-to-right', 0, 0),
            ('green-to-top', 1, 90),
            ('blue-to-left', 2, 180),
            ]:
        os.makedirs(os.path.join(folder, class_name))

        colors = np.linspace(200, 255, image_count)
        for i, color in enumerate(colors):
            pixel = [0, 0, 0]
            pixel[pixel_index] = color
            pil_img = _create_gradient_image(image_size, (0, 0, 0), pixel, rotation)
            img_path = os.path.join(class_name, str(i) + '.png')
            pil_img.save(os.path.join(folder, img_path))
            paths[class_name].append(img_path)

    return paths

def _create_gradient_image(size, color_from, color_to, rotation):
    """
    Make an image with a color gradient with a specific rotation
    """
    # create gradient
    rgb_arrays = [np.linspace(color_from[x], color_to[x], size).astype('uint8') for x in range(3)]
    gradient = np.concatenate(rgb_arrays)

    # extend to 2d
    picture = np.repeat(gradient, size)
    picture.shape = (3, size, size)

    # make image and rotate
    image = PIL.Image.fromarray(picture.T)
    image = image.rotate(rotation)

    return image


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Create-Imageset tool - DIGITS')

    ### Positional arguments

    parser.add_argument('folder',
            help='Where to save the images'
            )

    ### Optional arguments

    parser.add_argument('-s', '--image_size',
            type=int,
            help='Size of the images')
    parser.add_argument('-c', '--image_count',
            type=int,
            help='How many images')

    args = vars(parser.parse_args())

    print 'Creating images at "%s" ...' % args['folder']

    start_time = time.time()

    create_classification_imageset(args['folder'],
            image_size=args['image_size'],
            image_count=args['image_count'],
            )

    print 'Done after %s seconds' % (time.time() - start_time,)

