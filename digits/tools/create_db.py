#!/usr/bin/env python2
# Copyright (c) 2014-2017, NVIDIA CORPORATION.  All rights reserved.

import argparse
from collections import Counter
import logging
import math
import os
import Queue
import random
import re
import shutil
import sys
import threading
import time

# Find the best implementation available
try:
    from cStringIO import StringIO
except ImportError:
    from StringIO import StringIO

import h5py
import lmdb
import numpy as np
import PIL.Image

# Add path for DIGITS package
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import digits.config  # noqa
from digits import utils, log  # noqa

# Import digits.config first to set the path to Caffe
import caffe.io  # noqa
import caffe_pb2  # noqa

logger = logging.getLogger('digits.tools.create_db')


class Error(Exception):
    pass


class BadInputFileError(Error):
    """Input file is empty"""
    pass


class ParseLineError(Error):
    """Failed to parse a line in the input file"""
    pass


class LoadError(Error):
    """Failed to load image[s]"""
    pass


class WriteError(Error):
    """Failed to write image[s]"""
    pass


class Hdf5DatasetExtendError(Error):
    """Failed to extend an hdf5 dataset"""
    pass


class DbWriter(object):
    """
    Abstract class for writing to databases
    """

    def __init__(self, output_dir, image_height, image_width, image_channels):
        self._dir = output_dir
        os.makedirs(output_dir)
        self._image_height = image_height
        self._image_width = image_width
        self._image_channels = image_channels
        self._count = 0

    def write_batch(self, batch):
        raise NotImplementedError

    def count(self):
        return self._count


class LmdbWriter(DbWriter):
    # TODO
    pass


class Hdf5Writer(DbWriter):
    """
    A class for writing to HDF5 files
    """
    LIST_FILENAME = 'list.txt'
    DTYPE = 'float32'

    def __init__(self, **kwargs):
        """
        Keyword arguments:
        compression -- the type of dataset compression
        dset_limit -- the dataset size limit
        """
        self._compression = kwargs.pop('compression', None)
        self._dset_limit = kwargs.pop('dset_limit', None)
        super(Hdf5Writer, self).__init__(**kwargs)
        self._db = None

        if self._dset_limit is not None:
            self._max_count = self._dset_limit / (
                self._image_height * self._image_width * self._image_channels)
        else:
            self._max_count = None

    def write_batch(self, batch):
        # convert batch to numpy arrays
        if batch[0][0].ndim == 2:
            # add channel axis for grayscale images
            data_batch = np.array([i[0][..., np.newaxis] for i in batch])
        else:
            data_batch = np.array([i[0] for i in batch])
        # Transpose to (channels, height, width)
        data_batch = data_batch.transpose((0, 3, 1, 2))
        label_batch = np.array([i[1] for i in batch])

        # first batch
        if self._db is None:
            self._create_new_file(len(batch))
            self._db['data'][:] = data_batch
            self._db['label'][:] = label_batch
            self._count += len(batch)
            return

        current_count = self._db['data'].len()

        # will fit in current dataset
        if current_count + len(batch) <= self._max_count:
            self._db['data'].resize(current_count + len(batch), axis=0)
            self._db['label'].resize(current_count + len(batch), axis=0)
            self._db['data'][-len(batch):] = data_batch
            self._db['label'][-len(batch):] = label_batch
            self._count += len(batch)
            return

        # calculate how many will fit in current dataset
        split = self._max_count - current_count

        if split > 0:
            # put what we can into the current dataset
            self._db['data'].resize(self._max_count, axis=0)
            self._db['label'].resize(self._max_count, axis=0)
            self._db['data'][-split:] = data_batch[:split]
            self._db['label'][-split:] = label_batch[:split]
            self._count += split

        self._create_new_file(len(batch) - split)
        self._db['data'][:] = data_batch[split:]
        self._db['label'][:] = label_batch[split:]
        self._count += len(batch) - split

    def _create_new_file(self, initial_count):
        assert self._max_count is None or initial_count <= self._max_count, \
            'Your batch size is too large for your dataset limit - %d vs %d' % \
            (initial_count, self._max_count)

        # close the old file
        if self._db is not None:
            self._db.close()
            mode = 'a'
        else:
            mode = 'w'

        # get the filename
        filename = self._new_filename()
        logger.info('Creating HDF5 database at "%s" ...' %
                    os.path.join(*filename.split(os.sep)[-2:]))

        # update the list
        with open(self._list_filename(), mode) as outfile:
            outfile.write('%s\n' % filename)

        # create the new file
        self._db = h5py.File(os.path.join(self._dir, filename), 'w')

        # initialize the datasets
        self._db.create_dataset('data',
                                (initial_count, self._image_channels,
                                 self._image_height, self._image_width),
                                maxshape=(self._max_count, self._image_channels,
                                          self._image_height, self._image_width),
                                chunks=True, compression=self._compression, dtype=self.DTYPE)
        self._db.create_dataset('label',
                                (initial_count,),
                                maxshape=(self._max_count,),
                                chunks=True, compression=self._compression, dtype=self.DTYPE)

    def _list_filename(self):
        return os.path.join(self._dir, self.LIST_FILENAME)

    def _new_filename(self):
        return '%s.h5' % self.count()


def create_db(input_file, output_dir,
              image_width, image_height, image_channels,
              backend,
              resize_mode=None,
              image_folder=None,
              shuffle=True,
              mean_files=None,
              **kwargs):
    """
    Create a database of images from a list of image paths
    Raises exceptions on errors

    Arguments:
    input_file -- a textfile containing labelled image paths
    output_dir -- the location to store the created database
    image_width -- image resize width
    image_height -- image resize height
    image_channels -- image channels
    backend -- the DB format (lmdb/hdf5)

    Keyword arguments:
    resize_mode -- passed to utils.image.resize_image()
    shuffle -- if True, shuffle the images in the list before creating
    mean_files -- a list of mean files to save
    """
    # Validate arguments

    if not os.path.exists(input_file):
        raise ValueError('input_file does not exist')
    if os.path.exists(output_dir):
        logger.warning('removing existing database')
        if os.path.isdir(output_dir):
            shutil.rmtree(output_dir, ignore_errors=True)
        else:
            os.remove(output_dir)
    if image_width <= 0:
        raise ValueError('invalid image width')
    if image_height <= 0:
        raise ValueError('invalid image height')
    if image_channels not in [1, 3]:
        raise ValueError('invalid number of channels')
    if resize_mode not in [None, 'crop', 'squash', 'fill', 'half_crop']:
        raise ValueError('invalid resize_mode')
    if image_folder is not None and not os.path.exists(image_folder):
        raise ValueError('image_folder does not exist')
    if mean_files:
        for mean_file in mean_files:
            if os.path.exists(mean_file):
                logger.warning('overwriting existing mean file "%s"!' % mean_file)
            else:
                dirname = os.path.dirname(mean_file)
                if not dirname:
                    dirname = '.'
                if not os.path.exists(dirname):
                    raise ValueError('Cannot save mean file at "%s"' % mean_file)
    compute_mean = bool(mean_files)

    # Load lines from input_file into a load_queue

    load_queue = Queue.Queue()
    image_count = _fill_load_queue(input_file, load_queue, shuffle)

    # Start some load threads

    batch_size = _calculate_batch_size(image_count,
                                       bool(backend == 'hdf5'), kwargs.get('hdf5_dset_limit'),
                                       image_channels, image_height, image_width)
    num_threads = _calculate_num_threads(batch_size, shuffle)
    write_queue = Queue.Queue(2 * batch_size)
    summary_queue = Queue.Queue()

    for _ in xrange(num_threads):
        p = threading.Thread(target=_load_thread,
                             args=(load_queue, write_queue, summary_queue,
                                   image_width, image_height, image_channels,
                                   resize_mode, image_folder, compute_mean),
                             kwargs={'backend': backend,
                                     'encoding': kwargs.get('encoding', None)},
                             )
        p.daemon = True
        p.start()

    start = time.time()

    if backend == 'lmdb':
        _create_lmdb(image_count, write_queue, batch_size, output_dir,
                     summary_queue, num_threads,
                     mean_files, **kwargs)
    elif backend == 'hdf5':
        _create_hdf5(image_count, write_queue, batch_size, output_dir,
                     image_width, image_height, image_channels,
                     summary_queue, num_threads,
                     mean_files, **kwargs)
    else:
        raise ValueError('invalid backend')

    logger.info('Database created after %d seconds.' % (time.time() - start))


def _create_lmdb(image_count, write_queue, batch_size, output_dir,
                 summary_queue, num_threads,
                 mean_files=None,
                 encoding=None,
                 lmdb_map_size=None,
                 **kwargs):
    """
    Create an LMDB

    Keyword arguments:
    encoding -- image encoding format
    lmdb_map_size -- the initial LMDB map size
    """
    wait_time = time.time()
    threads_done = 0
    images_loaded = 0
    images_written = 0
    image_sum = None
    batch = []
    compute_mean = bool(mean_files)

    db = lmdb.open(output_dir,
                   map_size=lmdb_map_size,
                   map_async=True,
                   max_dbs=0)

    while (threads_done < num_threads) or not write_queue.empty():

        # Send update every 2 seconds
        if time.time() - wait_time > 2:
            logger.debug('Processed %d/%d' % (images_written, image_count))
            wait_time = time.time()

        processed_something = False

        if not summary_queue.empty():
            result_count, result_sum = summary_queue.get()
            images_loaded += result_count
            # Update total_image_sum
            if compute_mean and result_count > 0 and result_sum is not None:
                if image_sum is None:
                    image_sum = result_sum
                else:
                    image_sum += result_sum
            threads_done += 1
            processed_something = True

        if not write_queue.empty():
            datum = write_queue.get()
            batch.append(datum)

            if len(batch) == batch_size:
                _write_batch_lmdb(db, batch, images_written)
                images_written += len(batch)
                batch = []
            processed_something = True

        if not processed_something:
            time.sleep(0.2)

    if len(batch) > 0:
        _write_batch_lmdb(db, batch, images_written)
        images_written += len(batch)

    if images_loaded == 0:
        raise LoadError('no images loaded from input file')
    logger.debug('%s images loaded' % images_loaded)

    if images_written == 0:
        raise WriteError('no images written to database')
    logger.info('%s images written to database' % images_written)

    if compute_mean:
        _save_means(image_sum, images_written, mean_files)

    db.close()


def _create_hdf5(image_count, write_queue, batch_size, output_dir,
                 image_width, image_height, image_channels,
                 summary_queue, num_threads,
                 mean_files=None,
                 compression=None,
                 hdf5_dset_limit=None,
                 **kwargs):
    """
    Create an HDF5 file

    Keyword arguments:
    compression -- dataset compression format
    """
    wait_time = time.time()
    threads_done = 0
    images_loaded = 0
    images_written = 0
    image_sum = None
    batch = []
    compute_mean = bool(mean_files)

    writer = Hdf5Writer(
        output_dir=output_dir,
        image_height=image_height,
        image_width=image_width,
        image_channels=image_channels,
        dset_limit=hdf5_dset_limit,
        compression=compression,
    )

    while (threads_done < num_threads) or not write_queue.empty():

        # Send update every 2 seconds
        if time.time() - wait_time > 2:
            logger.debug('Processed %d/%d' % (images_written, image_count))
            wait_time = time.time()

        processed_something = False

        if not summary_queue.empty():
            result_count, result_sum = summary_queue.get()
            images_loaded += result_count
            # Update total_image_sum
            if compute_mean and result_count > 0 and result_sum is not None:
                if image_sum is None:
                    image_sum = result_sum
                else:
                    image_sum += result_sum
            threads_done += 1
            processed_something = True

        if not write_queue.empty():
            batch.append(write_queue.get())

            if len(batch) == batch_size:
                writer.write_batch(batch)
                images_written += len(batch)
                batch = []
            processed_something = True

        if not processed_something:
            time.sleep(0.2)

    if len(batch) > 0:
        writer.write_batch(batch)
        images_written += len(batch)

    assert images_written == writer.count()

    if images_loaded == 0:
        raise LoadError('no images loaded from input file')
    logger.debug('%s images loaded' % images_loaded)

    if images_written == 0:
        raise WriteError('no images written to database')
    logger.info('%s images written to database' % images_written)

    if compute_mean:
        _save_means(image_sum, images_written, mean_files)


def _fill_load_queue(filename, queue, shuffle):
    """
    Fill the queue with data from the input file
    Print the category distribution
    Returns the number of lines added to the queue

    NOTE: This can be slow on a large input file, but we need the total image
        count in order to report the progress, so we might as well read it all
    """
    total_lines = 0
    valid_lines = 0
    distribution = Counter()

    with open(filename) as infile:
        if shuffle:
            lines = infile.readlines()  # less memory efficient
            random.shuffle(lines)
            for line in lines:
                total_lines += 1
                try:
                    result = _parse_line(line, distribution)
                    valid_lines += 1
                    queue.put(result)
                except ParseLineError:
                    pass
        else:
            for line in infile:  # more memory efficient
                total_lines += 1
                try:
                    result = _parse_line(line, distribution)
                    valid_lines += 1
                    queue.put(result)
                except ParseLineError:
                    pass

    logger.debug('%s total lines in file' % total_lines)
    if valid_lines == 0:
        raise BadInputFileError('No valid lines in input file')
    logger.info('%s valid lines in file' % valid_lines)

    for key in sorted(distribution):
        logger.debug('Category %s has %d images.' % (key, distribution[key]))

    return valid_lines


def _parse_line(line, distribution):
    """
    Parse a line in the input file into (path, label)
    """
    line = line.strip()
    if not line:
        raise ParseLineError

    # Expect format - [/]path/to/file.jpg 123
    match = re.match(r'(.+)\s+(\d+)\s*$', line)
    if match is None:
        raise ParseLineError

    path = match.group(1)
    label = int(match.group(2))

    distribution[label] += 1

    return path, label


def _calculate_batch_size(image_count, is_hdf5=False, hdf5_dset_limit=None,
                          image_channels=None, image_height=None, image_width=None):
    """
    Calculates an appropriate batch size for creating this database
    """
    if is_hdf5 and hdf5_dset_limit is not None:
        return min(100, image_count, hdf5_dset_limit / (image_channels * image_height * image_width))
    else:
        return min(100, image_count)


def _calculate_num_threads(batch_size, shuffle):
    """
    Calculates an appropriate number of threads for creating this database
    """
    if shuffle:
        return min(10, int(round(math.sqrt(batch_size))))
    else:
        # XXX This is the only way to preserve order for now
        # This obviously hurts performance considerably
        return 1


def _load_thread(load_queue, write_queue, summary_queue,
                 image_width, image_height, image_channels,
                 resize_mode, image_folder, compute_mean,
                 backend=None, encoding=None):
    """
    Consumes items in load_queue
    Produces items to write_queue
    Stores cumulative results in summary_queue
    """
    images_added = 0
    if compute_mean:
        image_sum = _initial_image_sum(image_width, image_height, image_channels)
    else:
        image_sum = None

    while not load_queue.empty():
        try:
            path, label = load_queue.get(True, 0.05)
        except Queue.Empty:
            continue

        # prepend path with image_folder, if appropriate
        if not utils.is_url(path) and image_folder and not os.path.isabs(path):
            path = os.path.join(image_folder, path)

        try:
            image = utils.image.load_image(path)
        except utils.errors.LoadImageError as e:
            logger.warning('[%s] %s: %s' % (path, type(e).__name__, e))
            continue

        image = utils.image.resize_image(image,
                                         image_height, image_width,
                                         channels=image_channels,
                                         resize_mode=resize_mode,
                                         )

        if compute_mean:
            image_sum += image

        if backend == 'lmdb':
            datum = _array_to_datum(image, label, encoding)
            write_queue.put(datum)
        else:
            write_queue.put((image, label))

        images_added += 1

    summary_queue.put((images_added, image_sum))


def _initial_image_sum(width, height, channels):
    """
    Returns an array of zeros that will be used to store the accumulated sum of images
    """
    if channels == 1:
        return np.zeros((height, width), np.float64)
    else:
        return np.zeros((height, width, channels), np.float64)


def _array_to_datum(image, label, encoding):
    """
    Create a caffe Datum from a numpy.ndarray
    """
    if not encoding:
        # Transform to caffe's format requirements
        if image.ndim == 3:
            # Transpose to (channels, height, width)
            image = image.transpose((2, 0, 1))
            if image.shape[0] == 3:
                # channel swap
                # XXX see issue #59
                image = image[[2, 1, 0], ...]
        elif image.ndim == 2:
            # Add a channels axis
            image = image[np.newaxis, :, :]
        else:
            raise Exception('Image has unrecognized shape: "%s"' % image.shape)
        datum = caffe.io.array_to_datum(image, label)
    else:
        datum = caffe_pb2.Datum()
        if image.ndim == 3:
            datum.channels = image.shape[2]
        else:
            datum.channels = 1
        datum.height = image.shape[0]
        datum.width = image.shape[1]
        datum.label = label

        s = StringIO()
        if encoding == 'png':
            PIL.Image.fromarray(image).save(s, format='PNG')
        elif encoding == 'jpg':
            PIL.Image.fromarray(image).save(s, format='JPEG', quality=90)
        else:
            raise ValueError('Invalid encoding type')
        datum.data = s.getvalue()
        datum.encoded = True
    return datum


def _write_batch_lmdb(db, batch, image_count):
    """
    Write a batch to an LMDB database
    """
    try:
        with db.begin(write=True) as lmdb_txn:
            for i, datum in enumerate(batch):
                key = '%08d_%d' % (image_count + i, datum.label)
                lmdb_txn.put(key, datum.SerializeToString())

    except lmdb.MapFullError:
        # double the map_size
        curr_limit = db.info()['map_size']
        new_limit = curr_limit * 2
        try:
            db.set_mapsize(new_limit)  # double it
        except AttributeError as e:
            version = tuple(int(x) for x in lmdb.__version__.split('.'))
            if version < (0, 87):
                raise Error('py-lmdb is out of date (%s vs 0.87)' % lmdb.__version__)
            else:
                raise e
        # try again
        _write_batch_lmdb(db, batch, image_count)


def _save_means(image_sum, image_count, mean_files):
    """
    Save mean[s] to file
    """
    mean = np.around(image_sum / image_count).astype(np.uint8)
    for mean_file in mean_files:
        if mean_file.lower().endswith('.npy'):
            np.save(mean_file, mean)
        elif mean_file.lower().endswith('.binaryproto'):
            data = mean
            # Transform to caffe's format requirements
            if data.ndim == 3:
                # Transpose to (channels, height, width)
                data = data.transpose((2, 0, 1))
                if data.shape[0] == 3:
                    # channel swap
                    # XXX see issue #59
                    data = data[[2, 1, 0], ...]
            elif mean.ndim == 2:
                # Add a channels axis
                data = data[np.newaxis, :, :]

            blob = caffe_pb2.BlobProto()
            blob.num = 1
            blob.channels, blob.height, blob.width = data.shape
            blob.data.extend(data.astype(float).flat)

            with open(mean_file, 'wb') as outfile:
                outfile.write(blob.SerializeToString())
        elif mean_file.lower().endswith(('.jpg', '.jpeg', '.png')):
            image = PIL.Image.fromarray(mean)
            image.save(mean_file)
        else:
            logger.warning('Unrecognized file extension for mean file: "%s"' % mean_file)
            continue

        logger.info('Mean saved at "%s"' % mean_file)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Create-Db tool - DIGITS')

    # Positional arguments

    parser.add_argument('input_file',
                        help='An input file of labeled images')
    parser.add_argument('output_dir',
                        help='Path to the output database')
    parser.add_argument('width',
                        type=int,
                        help='width of resized images'
                        )
    parser.add_argument('height',
                        type=int,
                        help='height of resized images'
                        )

    # Optional arguments

    parser.add_argument('-c', '--channels',
                        type=int,
                        default=3,
                        help='channels of resized images (1 for grayscale, 3 for color [default])'
                        )
    parser.add_argument('-r', '--resize_mode',
                        help='resize mode for images (must be "crop", "squash" [default], "fill" or "half_crop")'
                        )
    parser.add_argument('-m', '--mean_file', action='append',
                        help="location to output the image mean (doesn't save mean if not specified)")
    parser.add_argument('-f', '--image_folder',
                        help='folder containing the images (if the paths in input_file are not absolute)')
    parser.add_argument('-s', '--shuffle',
                        action='store_true',
                        help='Shuffle images before saving'
                        )
    parser.add_argument('-e', '--encoding',
                        help='Image encoding format (jpg/png)'
                        )
    parser.add_argument('-C', '--compression',
                        help='Database compression format (gzip)'
                        )
    parser.add_argument('-b', '--backend',
                        default='lmdb',
                        help='The database backend - lmdb[default] or hdf5')
    parser.add_argument('--lmdb_map_size',
                        type=int,
                        help='The initial map size for LMDB (in MB)')
    parser.add_argument('--hdf5_dset_limit',
                        type=int,
                        default=2**31,
                        help='The size limit for HDF5 datasets')

    args = vars(parser.parse_args())

    if args['lmdb_map_size']:
        # convert from MB to B
        args['lmdb_map_size'] <<= 20

    try:
        create_db(args['input_file'], args['output_dir'],
                  args['width'], args['height'], args['channels'],
                  args['backend'],
                  resize_mode=args['resize_mode'],
                  image_folder=args['image_folder'],
                  shuffle=args['shuffle'],
                  mean_files=args['mean_file'],
                  encoding=args['encoding'],
                  compression=args['compression'],
                  lmdb_map_size=args['lmdb_map_size'],
                  hdf5_dset_limit=args['hdf5_dset_limit'],
                  )
    except Exception as e:
        logger.error('%s: %s' % (type(e).__name__, e.message))
        raise
