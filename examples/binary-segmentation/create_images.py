#!/usr/bin/env python2
# Copyright (c) 2016-2017, NVIDIA CORPORATION.  All rights reserved.
"""
Functions for creating a dummy image segmentation dataset
"""

import argparse
import numpy as np
import os
import PIL.Image
import PIL.ImageDraw
import shutil
import time


INPUT_FOLDER = "input"
TARGET_FOLDER = "target"


def create_images(folder, image_count, image_size, grid_size):
    """
    Create image pairs for segmentation dataset
    """
    # create folders
    if os.path.exists(folder):
        shutil.rmtree(folder)
    input_folder = os.path.join(folder, INPUT_FOLDER)
    os.makedirs(input_folder)
    target_folder = os.path.join(folder, TARGET_FOLDER)
    os.makedirs(target_folder)

    # create random x,y coordinates for image_count triangles
    coords = np.random.uniform(size=(image_count, 6)) * image_size

    for idx in xrange(image_count):
        triangle = coords[idx].tolist()
        # create blank images
        image_input = PIL.Image.new("L", (image_size, image_size), 255)
        image_target = PIL.Image.new("L", (image_size, image_size), 255)
        # draw an empty triangle
        draw = PIL.ImageDraw.Draw(image_input)
        draw.polygon(
            triangle,
            outline=0,
        )
        # draw a full triangle
        draw = PIL.ImageDraw.Draw(image_target)
        draw.polygon(
            triangle,
            outline=0,
            fill=0
        )
        # save images
        input_fname = os.path.join(input_folder, "%08d.png" % idx)
        target_fname = os.path.join(target_folder, "%08d.png" % idx)
        image_input.save(input_fname)
        image_target.save(target_fname)

    # create sample image grid
    image_grid = PIL.Image.new("L", (grid_size * image_size, grid_size * image_size), 255)
    coords = np.random.uniform(size=(grid_size, grid_size, 3, 2)) * image_size
    draw = PIL.ImageDraw.Draw(image_grid)
    for x in xrange(grid_size):
        for y in xrange(grid_size):
            triangle = coords[x][y]
            # shift
            triangle += np.array([x * image_size, y * image_size])
            triangle = triangle.reshape(6).tolist()
            # draw an empty triangle
            draw.polygon(
                triangle,
                outline=0,
            )
    image_grid.save(os.path.join(folder, "grid.png"))
    return


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Create segmentation image pairs')

    # Positional arguments
    parser.add_argument('output', help='Output folder')

    # Optional arguments
    parser.add_argument(
        '-c', '--image_count', type=int,
        default=10000, help='How many images to create')

    parser.add_argument(
        '-s',
        '--image_size',
        type=int,
        default=32,
        help='How many images to create')

    parser.add_argument(
        '-g',
        '--grid_size',
        type=int,
        default=10,
        help='Size of image grid in sample image')

    args = vars(parser.parse_args())

    start_time = time.time()

    create_images(args['output'], args['image_count'], args['image_size'], args['grid_size'])

    print 'Done after %s seconds' % (time.time() - start_time,)
