# Copyright (c) 2014-2016, NVIDIA CORPORATION.  All rights reserved.
from __future__ import absolute_import

import math
import os.path
import requests

# Find the best implementation available
try:
    from cStringIO import StringIO
except ImportError:
    from StringIO import StringIO

import matplotlib.colors
import matplotlib.cm
import numpy as np
import PIL.Image
import scipy.misc

from . import is_url, HTTP_TIMEOUT, errors

# Library defaults:
#   PIL.Image:
#       size -- (width, height)
#   np.array:
#       shape -- (height, width, channels)
#       range -- [0-255]
#       dtype -- uint8
#       channels -- RGB
#   caffe.datum:
#       datum.data type -- bytes (uint8)
#       datum.float_data type -- float32
#       when decoding images, channels are BGR
#   DIGITS:
#       image_dims -- (height, width, channels)

# List of supported file extensions
# Use like "if filename.endswith(SUPPORTED_EXTENSIONS)"
SUPPORTED_EXTENSIONS = ('.png','.jpg','.jpeg','.bmp','.ppm')

def load_image(path):
    """
    Reads a file from `path` and returns a PIL.Image with mode 'L' or 'RGB'
    Raises LoadImageError

    Arguments:
    path -- path to the image, can be a filesystem path or a URL
    """
    image = None
    if is_url(path):
        try:
            r = requests.get(path,
                    allow_redirects=False,
                    timeout=HTTP_TIMEOUT)
            r.raise_for_status()
            stream = StringIO(r.content)
            image = PIL.Image.open(stream)
        except requests.exceptions.RequestException as e:
            raise errors.LoadImageError, e.message
        except IOError as e:
            raise errors.LoadImageError, e.message
    elif os.path.exists(path):
        try:
            image = PIL.Image.open(path)
            image.load()
        except IOError as e:
            raise errors.LoadImageError, 'IOError: %s' % e.message
    else:
        raise errors.LoadImageError, '"%s" not found' % path

    if image.mode in ['L', 'RGB']:
        # No conversion necessary
        return image
    elif image.mode in ['1']:
        # Easy conversion to L
        return image.convert('L')
    elif image.mode in ['LA']:
        # Deal with transparencies
        new = PIL.Image.new('L', image.size, 255)
        new.paste(image, mask=image.convert('RGBA'))
        return new
    elif image.mode in ['CMYK', 'YCbCr']:
        # Easy conversion to RGB
        return image.convert('RGB')
    elif image.mode in ['P', 'RGBA']:
        # Deal with transparencies
        new = PIL.Image.new('RGB', image.size, (255, 255, 255))
        new.paste(image, mask=image.convert('RGBA'))
        return new
    else:
        raise errors.LoadImageError, 'Image mode "%s" not supported' % image.mode

def upscale(image, ratio):
    """
    return upscaled image array

    Arguments:
    image -- a (H,W,C) numpy.ndarray
    ratio -- scaling factor (>1)
    """
    if not isinstance(image, np.ndarray):
        raise ValueError('Expected ndarray')
    if ratio<1:
        raise ValueError('Ratio must be greater than 1 (ratio=%f)' % ratio)
    width = int(math.floor(image.shape[1] * ratio))
    height = int(math.floor(image.shape[0] * ratio))
    channels = image.shape[2]
    out = np.ndarray((height, width, channels),dtype=np.uint8)
    for x, y in np.ndindex((width,height)):
        out[y,x] = image[math.floor(y/ratio), math.floor(x/ratio)]
    return out

def resize_image(image, height, width,
        channels=None,
        resize_mode=None,
        ):
    """
    Resizes an image and returns it as a np.array

    Arguments:
    image -- a PIL.Image or numpy.ndarray
    height -- height of new image
    width -- width of new image

    Keyword Arguments:
    channels -- channels of new image (stays unchanged if not specified)
    resize_mode -- can be crop, squash, fill or half_crop
    """
    if resize_mode is None:
        resize_mode = 'squash'
    if resize_mode not in ['crop', 'squash', 'fill', 'half_crop']:
        raise ValueError('resize_mode "%s" not supported' % resize_mode)

    if channels not in [None, 1, 3]:
        raise ValueError('unsupported number of channels: %s' % channels)

    if isinstance(image, PIL.Image.Image):
        # Convert image mode (channels)
        if channels is None:
            image_mode = image.mode
            if image_mode == 'L':
                channels = 1
            elif image_mode == 'RGB':
                channels = 3
            else:
                raise ValueError('unknown image mode "%s"' % image_mode)
        elif channels == 1:
            # 8-bit pixels, black and white
            image_mode = 'L'
        elif channels == 3:
            # 3x8-bit pixels, true color
            image_mode = 'RGB'
        if image.mode != image_mode:
            image = image.convert(image_mode)
        image = np.array(image)
    elif isinstance(image, np.ndarray):
        if image.dtype != np.uint8:
            image = image.astype(np.uint8)
        if image.ndim == 3 and image.shape[2] == 1:
            image = image.reshape(image.shape[:2])
        if channels is None:
            if image.ndim == 2:
                channels = 1
            elif image.ndim == 3 and image.shape[2] == 3:
                channels = 3
            else:
                raise ValueError('invalid image shape: %s' % (image.shape,))
        elif channels == 1:
            if image.ndim != 2:
                if image.ndim == 3 and image.shape[2] == 3:
                    # color to grayscale
                    image = np.dot(image, [0.299, 0.587, 0.114]).astype(np.uint8)
                else:
                    raise ValueError('invalid image shape: %s' % (image.shape,))
        elif channels == 3:
            if image.ndim == 2:
                # grayscale to color
                image = np.repeat(image,3).reshape(image.shape + (3,))
            elif image.shape[2] != 3:
                raise ValueError('invalid image shape: %s' % (image.shape,))
    else:
        raise ValueError('resize_image() expected a PIL.Image.Image or a numpy.ndarray')

    # No need to resize
    if image.shape[0] == height and image.shape[1] == width:
        return image

    ### Resize
    interp = 'bilinear'

    width_ratio = float(image.shape[1]) / width
    height_ratio = float(image.shape[0]) / height
    if resize_mode == 'squash' or width_ratio == height_ratio:
        return scipy.misc.imresize(image, (height, width), interp=interp)
    elif resize_mode == 'crop':
        # resize to smallest of ratios (relatively larger image), keeping aspect ratio
        if width_ratio > height_ratio:
            resize_height = height
            resize_width = int(round(image.shape[1] / height_ratio))
        else:
            resize_width = width
            resize_height = int(round(image.shape[0] / width_ratio))
        image = scipy.misc.imresize(image, (resize_height, resize_width), interp=interp)

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
                resize_height = int(round(image.shape[0] / width_ratio))
                if (height - resize_height) % 2 == 1:
                    resize_height += 1
            else:
                resize_height = height
                resize_width = int(round(image.shape[1] / height_ratio))
                if (width - resize_width) % 2 == 1:
                    resize_width += 1
            image = scipy.misc.imresize(image, (resize_height, resize_width), interp=interp)
        elif resize_mode == 'half_crop':
            # resize to average ratio keeping aspect ratio
            new_ratio = (width_ratio + height_ratio) / 2.0
            resize_width = int(round(image.shape[1] / new_ratio))
            resize_height = int(round(image.shape[0] / new_ratio))
            if width_ratio > height_ratio and (height - resize_height) % 2 == 1:
                resize_height += 1
            elif width_ratio < height_ratio and (width - resize_width) % 2 == 1:
                resize_width += 1
            image = scipy.misc.imresize(image, (resize_height, resize_width), interp=interp)
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
    image -- a PIL.Image or np.ndarray
    """
    if image is None:
        return None
    elif isinstance(image, PIL.Image.Image):
        pass
    elif isinstance(image, np.ndarray):
        image = PIL.Image.fromarray(image)
    else:
        raise ValueError('image must be a PIL.Image or a np.ndarray')

    # Read format from the image
    fmt = image.format
    if not fmt:
        # default to JPEG
        fmt = 'jpeg'
    else:
        fmt = fmt.lower()

    string_buf = StringIO()
    image.save(string_buf, format=fmt)
    data = string_buf.getvalue().encode('base64').replace('\n', '')
    return 'data:image/%s;base64,%s' % (fmt, data)

def add_bboxes_to_image(image, bboxes, color='red', width=1):
    """
    Draw rectangles on the image for the bounding boxes
    Returns a PIL.Image

    Arguments:
    image -- input image
    bboxes -- bounding boxes in the [((l, t), (r, b)), ...] format

    Keyword arguments:
    color -- color to draw the rectangles
    width -- line width of the rectangles

    Example:
    image = Image.open(filename)
    add_bboxes_to_image(image, bboxes[filename], width=2, color='#FF7700')
    image.show()
    """
    def expanded_bbox(bbox, n):
        """
        Grow the bounding box by n pixels
        """
        l = min(bbox[0][0], bbox[1][0])
        r = max(bbox[0][0], bbox[1][0])
        t = min(bbox[0][1], bbox[1][1])
        b = max(bbox[0][1], bbox[1][1])
        return ((l - n, t - n), (r + n, b + n))

    from PIL import Image, ImageDraw
    draw = ImageDraw.Draw(image)
    for bbox in bboxes:
        for n in xrange(width):
            draw.rectangle(expanded_bbox(bbox, n), outline = color)

    return image

def get_layer_vis_square(data,
        allow_heatmap = True,
        normalize = True,
        min_img_dim = 100,
        max_width = 1200,
        channel_order = 'RGB',
        ):
    """
    Returns a vis_square for the given layer data

    Arguments:
    data -- a np.ndarray

    Keyword arguments:
    allow_heatmap -- if True, convert single channel images to heatmaps
    normalize -- whether to normalize the data when visualizing
    max_width -- maximum width for the vis_square
    """
    if channel_order not in ['RGB', 'BGR']:
        raise ValueError('Unsupported channel_order %s' % channel_order)
    if data.ndim == 1:
        # interpret as 1x1 grayscale images
        # (N, 1, 1)
        data = data[:, np.newaxis, np.newaxis]
    elif data.ndim == 2:
        # interpret as 1x1 grayscale images
        # (N, 1, 1)
        data = data.reshape((data.shape[0]*data.shape[1], 1, 1))
    elif data.ndim == 3:
        if data.shape[0] == 3:
            # interpret as a color image
            # (1, H, W,3)
            if channel_order == 'BGR':
                data = data[[2,1,0],...] # BGR to RGB (see issue #59)
            data = data.transpose(1,2,0)
            data = data[np.newaxis,...]
        else:
            # interpret as grayscale images
            # (N, H, W)
            pass
    elif data.ndim == 4:
        if data.shape[0] == 3:
            # interpret as HxW color images
            # (N, H, W, 3)
            data = data.transpose(1,2,3,0)
            if channel_order == 'BGR':
                data = data[:,:,:,[2,1,0]] # BGR to RGB (see issue #59)
        elif data.shape[1] == 3:
            # interpret as HxW color images
            # (N, H, W, 3)
            data = data.transpose(0,2,3,1)
            if channel_order == 'BGR':
                data = data[:,:,:,[2,1,0]] # BGR to RGB (see issue #59)
        else:
            # interpret as HxW grayscale images
            # (N, H, W)
            data = data.reshape((data.shape[0]*data.shape[1], data.shape[2], data.shape[3]))
    else:
        raise RuntimeError('unrecognized data shape: %s' % (data.shape,))

    # chop off data so that it will fit within max_width
    padsize = 0
    width = data.shape[2]
    if width > max_width:
        data = data[:1,:max_width,:max_width]
    else:
        if width > 1:
            padsize = 1
            width += 1
        n = max(max_width/width,1)
        n *= n
        data = data[:n]

    if not allow_heatmap and data.ndim == 3:
        data = data[...,np.newaxis]

    vis = vis_square(data,
            padsize     = padsize,
            normalize   = normalize,
            )

    # find minimum dimension and upscale if necessary
    _min = sorted(vis.shape[:2])[0]
    if _min < min_img_dim:
        # upscale image
        ratio = min_img_dim/float(_min)
        vis = upscale(vis, ratio)
    return vis

def vis_square(images,
        padsize=1,
        normalize=False,
        colormap=None,
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
    colormap -- a string representing one of the supported colormaps
            defaults to the "best" available
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
        # they're grayscale - convert to color
        color_images = np.ndarray(images.shape + (3,), dtype=images.dtype)
        if colormap is None:
            try:
                converter = matplotlib.cm.viridis
            except AttributeError:
                # old verison of matplotlib
                converter = matplotlib.cm.jet
        else:
            converter = matplotlib.cm.cmap_d[colormap]

        for i in xrange(len(images)):
            # convert and throw away alpha channel
            color_images[i] = converter(images[i]/255.0)[...,:3]
        images = 255.0 * color_images

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

    if tiles.shape[-1] == 1:
        # grayscale to color
        tiles = np.dstack([tiles.squeeze()] * 3)

    return tiles

