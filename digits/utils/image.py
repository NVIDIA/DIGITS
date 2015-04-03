# Copyright (c) 2014-2015, NVIDIA CORPORATION.  All rights reserved.

import os.path
import math

import requests
import cStringIO
import PIL.Image
import numpy as np
from scipy.ndimage.interpolation import zoom

from . import is_url, HTTP_TIMEOUT

# Library defaults:
#   PIL.Image:
#       size -- (width, height)
#   np.array:
#       shape -- (height, width, channels)
#       range -- [0-255]
#       dtype -- uint8
#       channels -- RGB
#   caffe:
#       shape -- (height, width, channels)
#       range -- [0-1]
#       dtype -- float32
#
# Within DIGITS:
#   When storing sizes, use (height, width, channels)

def load_image(path):
    """
    Reads a file from `path` and returns a PIL.Image

    Arguments:
    path -- path to the image, can be a filesystem path or a url
    """
    image = None
    try:
        if is_url(path):
            r = requests.get(path,
                    allow_redirects=False,
                    timeout=HTTP_TIMEOUT)
            if r.status_code == requests.codes.ok:
                stream = cStringIO.StringIO(r.content)
                image = PIL.Image.open(stream)
        elif os.path.exists(path):
            image = PIL.Image.open(path)
            image.load()
    except Exception as e:
        image = None

    return image

def resize_image(image, height, width,
        channels=None,
        resize_mode=None,
        ):
    """
    Resizes an image and returns it as a np.array

    Arguments:
    image -- a PIL.Image
    height -- height of new image
    width -- width of new image

    Keyword Arguments:
    channels -- channels of new image (stays unchanged if not specified)
    resize_mode -- can be crop, squash, fill or half_crop
    """
    assert isinstance(image, PIL.Image.Image), 'resize_image expected a PIL.Image'

    if resize_mode is None:
        resize_mode = 'half_crop'

    # Convert image mode (channels)
    if channels is None:
        image_mode = image.mode
        assert image_mode in ['L', 'RGB'], 'unknown image mode "%s"' % image_mode
        if image_mode == 'L':
            channels = 1
        elif image_mode == 'RGB':
            channels = 3
    elif channels == 1:
        # 8-bit pixels, black and white
        image_mode = 'L'
    elif channels == 3:
        # 3x8-bit pixels, true color
        image_mode = 'RGB'
    else:
        raise Exception('Unsupported number of channels: %s' % channels)
    if image.mode != image_mode:
        image = image.convert(image_mode)

    # No need to resize
    if image.size[0] == width and image.size[1] == height:
        return np.asarray(image)

    ### Resize

    width_ratio = float(image.size[0]) / width
    height_ratio = float(image.size[1]) / height
    if resize_mode == 'squash' or width_ratio == height_ratio:
        return np.array(image.resize((width, height), PIL.Image.ANTIALIAS))
    elif resize_mode == 'crop':
        # resize to smallest of ratios (relatively larger image), keeping aspect ratio
        if width_ratio > height_ratio:
            resize_height = height
            resize_width = int(round(image.size[0] / height_ratio))
        else:
            resize_width = width
            resize_height = int(round(image.size[1] / width_ratio))
        image = np.array(image.resize((resize_width, resize_height), PIL.Image.ANTIALIAS))

        # chop off ends of dimension that is still too long
        if width_ratio > height_ratio:
            start = int(round((resize_width-width)/2.0))
            return image[:,start:start+width]
        else:
            start = int(round((resize_height-height)/2.0))
            return image[start:start+height,:]
    else:
        if resize_mode == 'fill':
            # resize to biggest of ratios (relatively smaller image), keeping aspect ratio
            if width_ratio > height_ratio:
                resize_width = width
                resize_height = int(round(image.size[1] / width_ratio))
                if (height - resize_height) % 2 == 1:
                    resize_height += 1
            else:
                resize_height = height
                resize_width = int(round(image.size[0] / height_ratio))
                if (width - resize_width) % 2 == 1:
                    resize_width += 1
            image = np.array(image.resize((resize_width, resize_height), PIL.Image.ANTIALIAS))
        elif resize_mode == 'half_crop':
            # resize to average ratio keeping aspect ratio
            new_ratio = (width_ratio + height_ratio) / 2.0
            resize_width = int(round(image.size[0] / new_ratio))
            resize_height = int(round(image.size[1] / new_ratio))
            if width_ratio > height_ratio and (height - resize_height) % 2 == 1:
                resize_height += 1
            elif width_ratio < height_ratio and (width - resize_width) % 2 == 1:
                resize_width += 1
            image = np.array(image.resize((resize_width, resize_height), PIL.Image.ANTIALIAS))
            # chop off ends of dimension that is still too long
            if width_ratio > height_ratio:
                start = int(round((resize_width-width)/2.0))
                image = image[:,start:start+width]
            else:
                start = int(round((resize_height-height)/2.0))
                image = image[start:start+height,:]
        else:
            raise Exception('unrecognized resize_mode "%s"' % resize_mode)

        # fill ends of dimension that is too short with random noise
        if width_ratio > height_ratio:
            padding = (height - resize_height)/2
            noise_size = (padding, width)
            if channels > 1:
                noise_size += (channels,)
            noise = np.random.randint(0, 255, noise_size).astype('uint8')
            image = np.concatenate((noise, image, noise), axis=0)
        else:
            padding = (width - resize_width)/2
            noise_size = (height, padding)
            if channels > 1:
                noise_size += (channels,)
            noise = np.random.randint(0, 255, noise_size).astype('uint8')
            image = np.concatenate((noise, image, noise), axis=1)

        return image

def embed_image_html(image):
    """
    Returns an image embedded in HTML base64 format
    (Based on Caffe's web_demo)

    Arguments:
    image -- a PIL.Image or np.array
    """
    if image is None:
        return None
    if not isinstance(image, PIL.Image.Image):
        # assume it's a np.array
        image = PIL.Image.fromarray(image)
    string_buf = cStringIO.StringIO()
    image.save(string_buf, format='png')
    data = string_buf.getvalue().encode('base64').replace('\n', '')
    return 'data:image/png;base64,' + data

def vis_square(images,
        padsize=1,
        normalize=False,
        ):
    """
    Visualize each image in a grid of size approx sqrt(n) by sqrt(n)
    Returns a np.array image
    (Based on Caffe's filter_visualization notebook)

    Arguments:
    images -- an array of shape (n, height, width) or (n, height, width, channels)
            if channels is not set, a heatmap is computed for the result

    Keyword arguments:
    padsize -- how many pixels go inbetween the tiles
    normalize -- if true, scales (min, max) across all images out to (0, 1)
    """
    assert 3 <= images.ndim <= 4, 'images.ndim must be 3 or 4'
    # convert to float since we're going to do some math
    images = images.astype('float32')
    if normalize:
        images -= images.min()
        if images.max() > 0:
            images /= images.max()
            images *= 255

    if images.ndim == 3:
        # Compute the heatmap conversion
        blue    = 3 * images - 2*256-1
        green   = 3 * images - 1*256-1
        red     = 3 * images

        # Slap the channels back together
        images = np.concatenate( (red[...,np.newaxis], green[...,np.newaxis], blue[...,np.newaxis]), axis=3 )
        images = np.minimum(images,255)
        images = np.maximum(images,0)

    # convert back to uint8
    images = images.astype('uint8')

    # Compute the output image matrix dimensions
    n = int(np.ceil(np.sqrt(images.shape[0])))
    ny = n
    nx = n
    length = images.shape[0]
    if n*(n-1) >= length:
        nx = n-1

    # Add padding between the images
    padding = ((0, nx*ny - length), (0, padsize), (0, padsize)) + ((0, 0),) * (images.ndim - 3)
    padded = np.pad(images, padding, mode='constant', constant_values=255)

    # Tile the images beside each other
    tiles = padded.reshape( (ny, nx) + padded.shape[1:]).transpose( (0,2,1,3) + tuple(range(4, padded.ndim + 1)))
    tiles = tiles.reshape((ny * tiles.shape[1], nx * tiles.shape[3]) + tiles.shape[4:])

    return tiles

