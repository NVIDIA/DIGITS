#!/usr/bin/env python
# Copyright (c) 2014-2015, NVIDIA CORPORATION.  All rights reserved.

import sys
import os.path
import time
import argparse
import logging
from re import match as re_match
from shutil import rmtree
import random
import threading
import Queue

try:
    import digits
except ImportError:
    # Add path for DIGITS package
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import digits.config
digits.config.load_config()
from digits import utils, log

import numpy as np
import PIL.Image
import leveldb
import lmdb
from cStringIO import StringIO
# must call digits.config.load_config() before caffe to set the path
import caffe.io
try:
    import caffe_pb2
except ImportError:
    # See issue #32
    from caffe.proto import caffe_pb2

logger = logging.getLogger('digits.tools.create_db')

class DbCreator:
    """
    Creates a database for a neural network imageset
    """

    def __init__(self, db_path, backend='lmdb'):
        """
        Arguments:
        db_path -- where should the database be created

        Keyword arguments:
        backend -- 'lmdb' or 'leveldb'
        """
        # Can have trailing slash or not
        self.output_path = os.path.dirname(os.path.join(db_path, ''))
        self.name = os.path.basename(self.output_path)

        if os.path.exists(self.output_path):
            # caffe throws an error instead
            logger.warning('removing existing database %s' % self.output_path)
            rmtree(self.output_path, ignore_errors=True)

        if backend == 'lmdb':
            self.backend = 'lmdb'
            self.db = lmdb.open(self.output_path,
                    map_size=1000000000000, # ~1TB
                    map_async=True,
                    max_dbs=0)
        elif backend == 'leveldb':
            self.backend = 'leveldb'
            self.db = leveldb.LevelDB(self.output_path, error_if_exists=True)
        else:
            raise ValueError('unknown backend: "%s"' % backend)

        self.shutdown = threading.Event()
        self.keys_lock = threading.Lock()
        self.key_index = 0

    def create(self, input_file, width, height,
            channels    = 3,
            resize_mode = None,
            image_folder= None,
            shuffle     = True,
            mean_files  = None,
            encoding    = 'none',
            ):
        """
        Read an input file and create a database from the specified image/label pairs
        Returns True on success

        Arguments:
        input_file -- gives paths to images and their label (e.g. "path/to/image1.jpg 0\npath/to/image2.jpg 3")
        width -- width of resized images
        height -- width of resized images

        Keyword arguments:
        channels -- channels of resized images
        resize_mode -- can be crop, squash, fill or half_crop
        image_folder -- folder in which the images can be found
        shuffle -- shuffle images before saving
        mean_files -- an array of mean files to save (can be empty)
        encoding -- 'none', 'png' or 'jpg'
        """
        ### Validate input

        if not os.path.exists(input_file):
            logger.error('input_file does not exist')
            return False
        if height <= 0:
            logger.error('unsupported image height')
            return False
        self.height = height
        if width <= 0:
            logger.error('unsupported image width')
            return False
        self.width = width
        if channels not in [1,3]:
            logger.error('unsupported number of channels')
            return False
        self.channels = channels
        if resize_mode not in ['crop', 'squash', 'fill', 'half_crop']:
            logger.error('unsupported resize_mode')
            return False
        self.resize_mode = resize_mode
        if image_folder is not None and not os.path.exists(image_folder):
            logger.error('image_folder does not exist')
            return False
        self.image_folder = image_folder
        if mean_files:
            for mean_file in mean_files:
                if os.path.exists(mean_file):
                    logger.warning('overwriting existing mean file "%s"!' % mean_file)
                else:
                    dirname = os.path.dirname(mean_file)
                    if not dirname:
                        dirname = '.'
                    if not os.path.exists(dirname):
                        logger.error('Cannot save mean file at "%s"' % mean_file)
                        return False
        self.compute_mean = (mean_files and len(mean_files) > 0)
        if encoding not in ['none', 'png', 'jpg']:
            raise ValueError('Unsupported encoding format "%s"' % encoding)
        self.encoding = encoding

        ### Start working

        start = time.time()

        # TODO: adjust these values in real-time based on system load
        if not shuffle:
            #XXX This is the only way to preserve order for now
            # This obviously hurts performance considerably
            read_threads = 1
            write_threads = 1
        else:
            read_threads = 10
            write_threads = 10
        batch_size = 100

        total_images_added = 0
        total_image_sum = None

        # NOTE: The data could really stack up in these queues
        self.read_queue = Queue.Queue()
        self.write_queue = Queue.Queue(2*batch_size)

        # Tells read threads that if read_queue is empty, they can stop
        self.read_queue_built = threading.Event()
        self.read_thread_results = Queue.Queue()
        # Tells write threads that if write_queue is empty, they can stop
        self.write_queue_built = threading.Event()
        self.write_thread_results = Queue.Queue()

        # Read input_file and produce items to read_queue
        # NOTE This secion should be very efficient, because no progress about the job gets reported until after the read/write threads start
        lines_read = 0
        lines_per_category = {}
        with open(input_file, 'r') as f:
            lines = f.readlines()
            if shuffle:
                random.shuffle(lines)
            for line in lines:
                # Expect format - [/]path/to/file.jpg 123
                match = re_match(r'(.+)\s+(\d+)\s*$', line)
                if match is not None:
                    path = match.group(1)
                    label = int(match.group(2))
                    self.read_queue.put( (path, label) )
                    if label not in lines_per_category:
                        lines_per_category[label] = 1
                    else:
                        lines_per_category[label] += 1
                    lines_read += 1
        self.read_queue_built.set()

        if lines_read > 0:
            logger.info('Input images: %d' % lines_read)
        else:
            logger.error('no lines in input_file')
            return False

        for key in sorted(lines_per_category):
            logger.debug('Category %s has %d images.' % (key, lines_per_category[key]))

        # Start read threads
        for i in xrange(read_threads):
            p = threading.Thread(target=self.read_thread)
            p.daemon = True
            p.start()

        # Start write threads
        for i in xrange(write_threads):
            first_batch = int(batch_size * (i+1)/write_threads)
            p = threading.Thread(target=self.write_thread, args=(batch_size, first_batch))
            p.daemon = True
            p.start()

        # Wait for threads to finish
        wait_time = time.time()
        read_threads_done = 0
        write_threads_done = 0
        total_images_written = 0
        while write_threads_done < write_threads:
            if self.shutdown.is_set():
                # Die immediately
                return False

            # Send update every 2 seconds
            if time.time() - wait_time > 2:
                logger.debug('Processed %d/%d' % (lines_read - self.read_queue.qsize(), lines_read))
                #print '\tRead queue size: %d' % self.read_queue.qsize()
                #print '\tWrite queue size: %d' % self.write_queue.qsize()
                #print '\tRead threads done: %d' % read_threads_done
                #print '\tWrite threads done: %d' % write_threads_done
                wait_time = time.time()

            if not self.write_queue_built.is_set() and read_threads_done == read_threads:
                self.write_queue_built.set()

            while not self.read_thread_results.empty():
                images_added, image_sum = self.read_thread_results.get()
                total_images_added += images_added
                # Update total_image_sum
                if self.compute_mean and images_added > 0 and image_sum is not None:
                    if total_image_sum is None:
                        total_image_sum = image_sum
                    else:
                        total_image_sum += image_sum
                read_threads_done += 1

            while not self.write_thread_results.empty():
                result = self.write_thread_results.get()
                total_images_written += result
                write_threads_done += 1

            try:
                time.sleep(0.2)
            except KeyboardInterrupt:
                self.shutdown.set()
                return False

        if total_images_added == 0:
            logger.error('no images added')
            return False

        # Compute image mean
        if self.compute_mean and total_image_sum is not None:
            mean = np.around(total_image_sum / total_images_added).astype(np.uint8)
            for mean_file in mean_files:
                if mean_file.lower().endswith('.npy'):
                    np.save(mean_file, mean)
                elif mean_file.lower().endswith('.binaryproto'):
                    data = mean
                    # Transform to caffe's format requirements
                    if data.ndim == 3:
                        # Transpose to (channels, height, width)
                        data = data.transpose((2,0,1))
                        if data.shape[0] == 3:
                            # channel swap
                            # XXX see issue #59
                            data = data[[2,1,0],...]
                    elif mean.ndim == 2:
                        # Add a channels axis
                        data = data[np.newaxis,:,:]

                    blob = caffe_pb2.BlobProto()
                    blob.num = 1
                    blob.channels, blob.height, blob.width = data.shape
                    blob.data.extend(data.astype(float).flat)

                    with open(mean_file, 'w') as outfile:
                        outfile.write(blob.SerializeToString())
                elif mean_file.lower().endswith(('.jpg', '.jpeg', '.png')):
                    image = PIL.Image.fromarray(mean)
                    image.save(mean_file)
                else:
                    logger.warning('Unrecognized file extension for mean file: "%s"' % mean_file)
                    continue

                logger.info('Mean saved at "%s"' % mean_file)

        logger.info('Database created after %d seconds.' % (time.time() - start))
        logger.info('Total images added: %d' % total_images_written)

        self.shutdown.set()
        return True


    def read_thread(self):
        """
        Consumes items in read_queue which are lines from input_file
        Produces items to write_queue which are Datums
        """
        images_added = 0
        image_sum = self.initial_image_sum()

        while not self.read_queue_built.is_set() or not self.read_queue.empty():

            if self.shutdown.is_set():
                # Die immediately
                return

            try:
                path, label = self.read_queue.get(True, 0.05)
            except Queue.Empty:
                continue

            try:
                datum = self.path_to_datum(path, label, image_sum)
                if datum is not None:
                    self.write_queue.put(datum)
                    images_added += 1
            except Exception as e:
                # This could be a ton of warnings
                logger.warning('DbCreator.read_thread caught %s: %s' % (type(e).__name__, e) )
                # TODO: count number of errors and abort if too many encountered

        self.read_thread_results.put( (images_added, image_sum) )
        return True

    def initial_image_sum(self):
        """
        Returns an array of zeros that will be used to store the accumulated sum of images
        """
        if self.compute_mean:
            if self.channels == 1:
                return np.zeros((self.height, self.width), np.float64)
            else:
                return np.zeros((self.height, self.width, self.channels), np.float64)
        else:
            return None

    def path_to_datum(self, path, label,
            image_sum = None):
        """
        Creates a Datum from a path and a label
        May also update image_sum, if computing mean

        Arguments:
        path -- path to the image (filesystem path or URL)
        label -- numeric label for this image's category

        Keyword arguments:
        image_sum -- numpy array that stores a running sum of added images
        """
        # prepend path with image_folder, if appropriate
        if not utils.is_url(path) and self.image_folder and not os.path.isabs(path):
            path = os.path.join(self.image_folder, path)

        image = utils.image.load_image(path)
        image = utils.image.resize_image(image,
                self.height, self.width,
                channels    = self.channels,
                resize_mode = self.resize_mode,
                )

        if self.compute_mean and image_sum is not None:
            image_sum += image

        if not self.encoding or self.encoding == 'none':
            # Transform to caffe's format requirements
            if image.ndim == 3:
                # Transpose to (channels, height, width)
                image = image.transpose((2,0,1))
                if image.shape[0] == 3:
                    # channel swap
                    # XXX see issue #59
                    image = image[[2,1,0],...]
            elif image.ndim == 2:
                # Add a channels axis
                image = image[np.newaxis,:,:]
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
            if self.encoding == 'png':
                PIL.Image.fromarray(image).save(s, format='PNG')
            elif self.encoding == 'jpg':
                PIL.Image.fromarray(image).save(s, format='JPEG', quality=90)
            datum.data = s.getvalue()
            datum.encoded = True
        return datum

    def write_thread(self, batch_size, batch_extra):
        """
        Consumes items in write_queue which are Datums
        Writes the image data to the database in batches

        Arguments:
        batch_size -- how many records to add to the database at a time
        batch_extra -- how many extra entries to include with the first batch (used for staging write batches)
        """
        if not batch_size > 0:
            logger.error('batch_size must be positive')
            return False

        batch = []

        images_added = 0
        while not self.write_queue_built.is_set() or not self.write_queue.empty():
            if self.shutdown.is_set():
                # Die immediately
                return

            try:
                datum = self.write_queue.get(True, 0.05)
            except Queue.Empty:
                continue

            batch.append(datum)
            images_added += 1
            if (batch_extra and len(batch) == batch_extra) or (len(batch) == batch_size):
                self.write_batch(batch)
                batch_extra = 0
                batch = []

        # Write last batch
        if len(batch):
            self.write_batch(batch)

        self.write_thread_results.put(images_added)
        return True

    def write_batch(self, batch):
        """
        Write a batch to the database

        Arguments:
        batch -- an array of Datums
        """
        keys = self.get_keys(len(batch))
        if self.backend == 'lmdb':
            lmdb_txn = self.db.begin(write=True)
            for i, datum in enumerate(batch):
                lmdb_txn.put('%08d_%d' % (keys[i], datum.label), datum.SerializeToString())
            lmdb_txn.commit()
        elif self.backend == 'leveldb':
            leveldb_batch = leveldb.WriteBatch()
            for i, datum in enumerate(batch):
                leveldb_batch.Put('%08d_%d' % (keys[i], datum.label), datum.SerializeToString())
            self.db.Write(leveldb_batch)
        else:
            logger.error('unsupported backend')
            return False

    def get_keys(self, num):
        """
        Return a range of keys to be used for a write batch

        Arguments:
        num -- how many keys
        """
        i = None
        self.keys_lock.acquire()
        try:
            i = self.key_index
            self.key_index += num
        finally:
            self.keys_lock.release()
        return range(i, i+num)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Create-Db tool - DIGITS')

    ### Positional arguments

    parser.add_argument('input_file',
            help='An input file of labeled images')
    parser.add_argument('db_name',
            help='Path to the output database')
    parser.add_argument('width',
            type=int,
            help='width of resized images'
            )
    parser.add_argument('height',
            type=int,
            help='height of resized images'
            )

    ### Optional arguments

    parser.add_argument('-c', '--channels',
            type=int,
            default=3,
            help='channels of resized images (1 for grayscale, 3 for color [default])'
            )
    parser.add_argument('-r', '--resize_mode',
            default='squash',
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
    parser.add_argument('-b', '--backend',
            default='lmdb',
            help='db backend [default=lmdb]'
            )
    parser.add_argument('-e', '--encoding',
            default = 'none',
            help = 'Choose encoding format ("jpg", "png" or "none" [default])'
            )

    args = vars(parser.parse_args())

    db = DbCreator(args['db_name'],
            backend=args['backend'])

    if db.create(args['input_file'], args['width'], args['height'],
            channels        = args['channels'],
            resize_mode     = args['resize_mode'],
            image_folder    = args['image_folder'],
            shuffle         = args['shuffle'],
            mean_files      = args['mean_file'],
            encoding        = args['encoding'],
            ):
        sys.exit(0)
    else:
        sys.exit(1)

