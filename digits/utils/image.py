# Copyright (c) 2014-2015, NVIDIA CORPORATION.  All rights reserved.

import os.path
import math

import requests
import cStringIO
import PIL.Image
import numpy as np

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
        colormap='jet',
        ):
    """
    Visualize each image in a grid of size approx sqrt(n) by sqrt(n)
    Returns a np.array image
    (Based on Caffe's filter_visualization notebook)

    Arguments:
    images -- an array of shape (N, H, W) or (N, H, W, C)
            if C is not set, a heatmap is computed for the result

    Keyword arguments:
    padsize -- how many pixels go inbetween the tiles
    normalize -- if true, scales (min, max) across all images out to (0, 1)
    colormap -- a string representing one of the suppoted colormaps
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
        # they're grayscale - convert to a colormap
        redmap, greenmap, bluemap = get_color_map(colormap)

        red = np.interp(images*(len(redmap)-1)/255.0, xrange(len(redmap)), redmap)
        green = np.interp(images*(len(greenmap)-1)/255.0, xrange(len(greenmap)), greenmap)
        blue = np.interp(images*(len(bluemap)-1)/255.0, xrange(len(bluemap)), bluemap)

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

def get_color_map(name):
    """
    Return a colormap as (redmap, greenmap, bluemap)

    Arguments:
    name -- the name of the colormap. If unrecognized, will default to 'jet'.
    """
    redmap = [0]
    greenmap = [0]
    bluemap = [0]
    if name == 'white':
        # essentially a noop
        redmap      = [0,1]
        greenmap    = [0,1]
        bluemap     = [0,1]
    elif name == 'simple':
        redmap      = [0,1,1,1]
        greenmap    = [0,0,1,1]
        bluemap     = [0,0,0,1]
    elif name == 'hot':
        redmap = [0, 0.03968253968253968, 0.07936507936507936, 0.119047619047619, 0.1587301587301587, 0.1984126984126984, 0.2380952380952381, 0.2777777777777778, 0.3174603174603174, 0.3571428571428571, 0.3968253968253968, 0.4365079365079365, 0.4761904761904762, 0.5158730158730158, 0.5555555555555556, 0.5952380952380952, 0.6349206349206349, 0.6746031746031745, 0.7142857142857142, 0.753968253968254, 0.7936507936507936, 0.8333333333333333, 0.873015873015873, 0.9126984126984127, 0.9523809523809523, 0.992063492063492, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
        greenmap = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.03174603174603163, 0.0714285714285714, 0.1111111111111112, 0.1507936507936507, 0.1904761904761905, 0.23015873015873, 0.2698412698412698, 0.3095238095238093, 0.3492063492063491, 0.3888888888888888, 0.4285714285714284, 0.4682539682539679, 0.5079365079365079, 0.5476190476190477, 0.5873015873015872, 0.6269841269841268, 0.6666666666666665, 0.7063492063492065, 0.746031746031746, 0.7857142857142856, 0.8253968253968254, 0.8650793650793651, 0.9047619047619047, 0.9444444444444442, 0.984126984126984, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
        bluemap = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.04761904761904745, 0.1269841269841265, 0.2063492063492056, 0.2857142857142856, 0.3650793650793656, 0.4444444444444446, 0.5238095238095237, 0.6031746031746028, 0.6825396825396828, 0.7619047619047619, 0.8412698412698409, 0.92063492063492, 1]
    elif name == 'rainbow':
        redmap = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0.9365079365079367, 0.8571428571428572, 0.7777777777777777, 0.6984126984126986, 0.6190476190476191, 0.53968253968254, 0.4603174603174605, 0.3809523809523814, 0.3015873015873018, 0.2222222222222223, 0.1428571428571432, 0.06349206349206415, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.03174603174603208, 0.08465608465608465, 0.1375661375661377, 0.1904761904761907, 0.2433862433862437, 0.2962962962962963, 0.3492063492063493, 0.4021164021164023, 0.4550264550264553, 0.5079365079365079, 0.5608465608465609, 0.6137566137566139, 0.666666666666667]
        greenmap = [0, 0.03968253968253968, 0.07936507936507936, 0.119047619047619, 0.1587301587301587, 0.1984126984126984, 0.2380952380952381, 0.2777777777777778, 0.3174603174603174, 0.3571428571428571, 0.3968253968253968, 0.4365079365079365, 0.4761904761904762, 0.5158730158730158, 0.5555555555555556, 0.5952380952380952, 0.6349206349206349, 0.6746031746031745, 0.7142857142857142, 0.753968253968254, 0.7936507936507936, 0.8333333333333333, 0.873015873015873, 0.9126984126984127, 0.9523809523809523, 0.992063492063492, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0.9841269841269842, 0.9047619047619047, 0.8253968253968256, 0.7460317460317465, 0.666666666666667, 0.587301587301587, 0.5079365079365079, 0.4285714285714288, 0.3492063492063493, 0.2698412698412698, 0.1904761904761907, 0.1111111111111116, 0.03174603174603208, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        bluemap = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.01587301587301582, 0.09523809523809534, 0.1746031746031744, 0.2539682539682535, 0.333333333333333, 0.412698412698413, 0.4920634920634921, 0.5714285714285712, 0.6507936507936507, 0.7301587301587302, 0.8095238095238093, 0.8888888888888884, 0.9682539682539679, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
    elif name == 'winter':
        greenmap = [0, 1]
        bluemap = [1, 0.5]
    else:
        if name != 'jet':
            print 'Warning: colormap "%s" not supported. Using jet instead.' % name
        redmap      = [0,0,0,0,0.5,1,1,1,0.5]
        greenmap    = [0,0,0.5,1,1,1,0.5,0,0]
        bluemap     = [0.5,1,1,1,0.5,0,0,0,0]
    return 255.0 * np.array(redmap), 255.0 * np.array(greenmap), 255.0 * np.array(bluemap)
