#!/usr/bin/env python
# Copyright (c) 2015, NVIDIA CORPORATION.  All rights reserved.
"""
Functions for creating temporary LMDBs
Used in test_views
"""

import os
import sys
import time
import argparse
from collections import defaultdict
from cStringIO import StringIO

import numpy as np
import PIL.Image
import lmdb

try:
    import caffe_pb2
except ImportError:
    # See issue #32
    from caffe.proto import caffe_pb2


IMAGE_SIZE  = 10
TRAIN_IMAGE_COUNT = 100
VAL_IMAGE_COUNT = 20


def create_lmdbs(folder, image_width=None, image_height=None, image_count=None):
    """
    Creates LMDBs for generic inference
    Returns the filename for a test image

    Creates these files in "folder":
        train_images/
        train_labels/
        val_images/
        val_labels/
        mean.binaryproto
        test.png
    """
    if image_width is None:
        image_width = IMAGE_SIZE
    if image_height is None:
        image_height = IMAGE_SIZE

    if image_count is None:
        train_image_count = TRAIN_IMAGE_COUNT
    else:
        train_image_count = image_count
    val_image_count = VAL_IMAGE_COUNT

    # Used to calculate the gradients later
    yy, xx = np.mgrid[:image_height, :image_width].astype('float')

    for phase, image_count in [
            ('train', train_image_count),
            ('val', val_image_count)]:
        image_db = lmdb.open(os.path.join(folder, '%s_images' % phase),
                map_size=1024**4, # 1TB
                map_async=True,
                max_dbs=0)
        label_db = lmdb.open(os.path.join(folder, '%s_labels' % phase),
                map_size=1024**4, # 1TB
                map_async=True,
                max_dbs=0)

        write_batch_size = 10

        image_txn = image_db.begin(write=True)
        label_txn = label_db.begin(write=True)

        image_sum = np.zeros((image_height, image_width), 'float64')

        for i in xrange(image_count):
            xslope, yslope = np.random.random_sample(2) - 0.5
            a = xslope * 255 / image_width
            b = yslope * 255 / image_height
            image = a * (xx - image_width/2) + b * (yy - image_height/2) + 127.5

            image_sum += image
            image = image.astype('uint8')

            pil_img = PIL.Image.fromarray(image)
            #pil_img.save(os.path.join(folder, '%s_%d.png' % (phase, i)))

            # create image Datum
            image_datum = caffe_pb2.Datum()
            image_datum.height = image.shape[0]
            image_datum.width = image.shape[1]
            image_datum.channels = 1
            s = StringIO()
            pil_img.save(s, format='PNG')
            image_datum.data = s.getvalue()
            image_datum.encoded = True
            image_txn.put(str(i), image_datum.SerializeToString())

            # create label Datum
            label_datum = caffe_pb2.Datum()
            label_datum.channels, label_datum.height, label_datum.width = 1, 1, 2
            label_datum.float_data.extend(np.array([xslope, yslope]).flat)
            label_txn.put(str(i), label_datum.SerializeToString())

            if ((i+1)%write_batch_size) == 0:
                image_txn.commit()
                image_txn = image_db.begin(write=True)
                label_txn.commit()
                label_txn = label_db.begin(write=True)

        # close databases
        image_db.close()
        label_db.close()

        # save mean
        mean_image = (image_sum / image_count).astype('uint8')
        _save_mean(mean_image, os.path.join(folder, '%s_mean.png' % phase))
        _save_mean(mean_image, os.path.join(folder, '%s_mean.binaryproto' % phase))

    # create test image
    #   The network should be able to easily produce two numbers >1
    xslope, yslope = 0.5, 0.5
    a = xslope * 255 / image_width
    b = yslope * 255 / image_height
    test_image = a * (xx - image_width/2) + b * (yy - image_height/2) + 127.5
    test_image = test_image.astype('uint8')
    pil_img = PIL.Image.fromarray(test_image)
    test_image_filename = os.path.join(folder, 'test.png')
    pil_img.save(test_image_filename)

    return test_image_filename

def _save_mean(mean, filename):
    """
    Saves mean to file

    Arguments:
    mean -- the mean as an np.ndarray
    filename -- the location to save the image
    """
    if filename.endswith('.binaryproto'):
        blob = caffe_pb2.BlobProto()
        blob.num = 1
        blob.channels = 1
        blob.height, blob.width = mean.shape
        blob.data.extend(mean.astype(float).flat)
        with open(filename, 'w') as outfile:
            outfile.write(blob.SerializeToString())

    elif filename.endswith(('.jpg', '.jpeg', '.png')):
        image = PIL.Image.fromarray(mean)
        image.save(filename)
    else:
        raise ValueError('unrecognized file extension')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Create-LMDB tool - DIGITS')

    ### Positional arguments

    parser.add_argument('folder',
            help='Where to save the images'
            )

    ### Optional arguments

    parser.add_argument('-x', '--image_width',
            type=int,
            help='Width of the images')
    parser.add_argument('-y', '--image_height',
            type=int,
            help='Height of the images')
    parser.add_argument('-c', '--image_count',
            type=int,
            help='How many images')

    args = vars(parser.parse_args())

    if os.path.exists(args['folder']):
        print 'ERROR: Folder already exists'
        sys.exit(1)
    else:
        os.makedirs(args['folder'])

    print 'Creating images at "%s" ...' % args['folder']

    start_time = time.time()

    create_lmdbs(args['folder'],
            image_width=args['image_width'],
            image_height=args['image_height'],
            image_count=args['image_count'],
            )

    print 'Done after %s seconds' % (time.time() - start_time,)

