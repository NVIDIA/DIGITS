#!/usr/bin/env python2
# Copyright (c) 2016, NVIDIA CORPORATION.  All rights reserved.

import argparse
# Find the best implementation available
try:
    from cStringIO import StringIO
except ImportError:
    from StringIO import StringIO
import lmdb
import logging
import numpy as np
import os
import PIL.Image
import Queue
import sys
import threading

# Add path for DIGITS package
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import digits.config
digits.config.load_config()
from digits import extensions, log
from digits.job import Job

# Run load_config() first to set the path to Caffe
import caffe.io
import caffe_pb2

logger = logging.getLogger('digits.tools.create_dataset')


class DbWriter(threading.Thread):
    """
    Abstract class for writing to databases
    """

    def __init__(self, output_dir, total_records=None):
        self._dir = output_dir
        self.write_queue = Queue.Queue(10)
        # sequence number
        self.seqn = 0
        self.total_records = total_records
        self.done = False
        threading.Thread.__init__(self)

    def write_batch_threadsafe(self, batch):
        """
        This function writes a batch of data into the database
        This may be called from multiple threads
        """
        self.write_queue.put(batch)

    def set_done(self):
        """
        Instructs writer thread to complete after queue becomes empty
        """
        self.done = True

    def run(self):
        """
        DB Writer thread entry point
        """
        while True:
            try:
                batch = self.write_queue.get(timeout=0.1)
            except Queue.Empty:
                if self.done:
                    # break out of main loop and terminate
                    break
                else:
                    # just keep looping
                    continue
            self.write_batch_threadunsafe(batch)


class LmdbWriter(DbWriter):

    def __init__(self,
                 dataset_dir,
                 stage,
                 feature_encoding,
                 label_encoding,
                 **kwargs):
        self.stage = stage
        db_dir = os.path.join(dataset_dir, stage)
        if not os.path.exists(db_dir):
            os.makedirs(db_dir)
        super(LmdbWriter, self).__init__(dataset_dir, **kwargs)

        # create LMDB for features
        self.feature_db = self.create_lmdb("features")
        # will create LMDB for labels later if necessary
        self.label_db = None
        # encoding
        self.feature_encoding = feature_encoding
        self.label_encoding = label_encoding

    def create_lmdb(self, db_type):
        sub_dir = os.path.join(self.stage, db_type)
        db_dir = os.path.join(self._dir, sub_dir)
        db = lmdb.open(
            db_dir,
            map_async=True,
            max_dbs=0)
        logger.info('Created %s db for stage %s in %s' % (db_type,
                                                          self.stage,
                                                          sub_dir))
        return db

    def array_to_datum(self, data, scalar_label, encoding):
        if data.ndim != 3:
            raise ValueError('Invalid number of dimensions: %d' % data.ndim)
        if data.shape[0] == 3:
            # RGB to BGR
            # XXX see issue #59
            data = data[[2, 1, 0], ...]
        if encoding == 'none':
            datum = caffe.io.array_to_datum(data, scalar_label)
        else:
            # Transpose to (height, width, channel)
            data = data.transpose((1, 2, 0))
            datum = caffe_pb2.Datum()
            datum.height = data.shape[0]
            datum.width = data.shape[1]
            datum.channels = data.shape[2]
            datum.label = scalar_label
            if data.shape[2] == 1:
                # grayscale
                data = data[:, :, 0]
            s = StringIO()
            if encoding == 'png':
                PIL.Image.fromarray(data).save(s, format='PNG')
            elif encoding == 'jpg':
                PIL.Image.fromarray(data).save(s, format='JPEG', quality=90)
            else:
                raise ValueError('Invalid encoding type')
            datum.data = s.getvalue()
            datum.encoded = True
        return datum

    def write_batch(self, batch):
        """
        encode data into datum objects
        this may be called from multiple encoder threads
        """
        datums = []
        for (feature, label) in batch:
            # restrict features to 3D data (Caffe Datum objects)
            assert feature.ndim == 3, "LMDB/Caffe expect 3D data"
            # restrict labels to 3D data (Caffe Datum objects) or scalars
            assert label.ndim == 3 or label.size == 1, "LMDB/Caffe expect 3D or scalar label"
            if label.size > 1:
                label_datum = self.array_to_datum(
                    label,
                    0,
                    self.label_encoding)
                # setting label to 0 - it will be unused as there is
                # a dedicated label DB
                label = 0
            else:
                label = label[0]
                label_datum = None
            feature_datum = self.array_to_datum(
                feature,
                label,
                self.feature_encoding)
            datums.append(
                (feature_datum.SerializeToString(),
                 label_datum.SerializeToString() if label_datum else None))
        self.write_batch_threadsafe(datums)

    def write_batch_threadunsafe(self, batch):
        """
        Write batch do DB, this must only be called from the writer thread
        """
        feature_datums = []
        label_datums = []
        for (feature, label) in batch:
            key = "%09d" % self.seqn
            if label is not None:
                if self.label_db is None:
                    self.label_db = self.create_lmdb("labels")
                label_datums.append((key, label))
            feature_datums.append((key, feature))
            self.seqn += 1
        self.write_datums(self.feature_db, feature_datums)
        if len(label_datums) > 0:
            self.write_datums(self.label_db, label_datums)
        logger.info('Processed %d/%d' % (self.seqn, self.total_records))

    def write_datums(self, db, batch):
        try:
            with db.begin(write=True) as lmdb_txn:
                for key, datum in batch:
                    lmdb_txn.put(key, datum)
        except lmdb.MapFullError:
            # double the map_size
            curr_limit = db.info()['map_size']
            new_limit = curr_limit*2
            logger.info(
                'Doubling LMDB map size to %sMB ...' % (new_limit >> 20,))
            try:
                db.set_mapsize(new_limit)  # double it
            except AttributeError as e:
                version = tuple(int(x) for x in lmdb.__version__.split('.'))
                if version < (0, 87):
                    raise ValueError('py-lmdb is out of date (%s vs 0.87)' % lmdb.__version__)
                else:
                    raise e
            # try again
            self.write_datums(db, batch)


class Encoder(threading.Thread):
    def __init__(self, queue, writer, extension, error_queue):
        self.extension = extension
        self.queue = queue
        self.writer = writer
        self.label_shape = None
        self.feature_shape = None
        self.feature_sum = None
        self.processed_count = 0
        self.error_queue = error_queue
        threading.Thread.__init__(self)

    def run(self):
        data = []
        while True:
            # get entry ID
            # don't block- if the queue is empty then we're done
            try:
                batch = self.queue.get_nowait()
            except Queue.Empty:
                # break out of main loop and terminate
                break

            try:
                data = []
                for entry_id in batch:
                    # call into extension to format entry into number arrays
                    feature, label = self.extension.encode_entry(entry_id)

                    # check feature and label shapes
                    if self.feature_shape is None:
                        self.feature_shape = feature.shape
                        self.feature_sum = np.zeros(self.feature_shape, np.float64)
                    else:
                        assert self.feature_shape == feature.shape
                    if self.label_shape is None:
                        self.label_shape = label.shape
                    else:
                        assert self.label_shape == label.shape

                    # accumulate sum for mean file calculation
                    self.feature_sum += feature

                    # aggregate data
                    data.append((feature, label))

                    self.processed_count += 1

                if len(data) >= 0:
                    # write data
                    self.writer.write_batch(data)
            except Exception as e:
                self.error_queue.put('%s: %s' % (type(e).__name__, e.message))
                raise

class DbCreator(object):

    def create_db(self, extension, stage, dataset_dir, batch_size, num_threads, feature_encoding, label_encoding):
        # retrieve itemized list of entries
        entry_ids = extension.itemize_entries(stage)
        entry_count = len(entry_ids)

        if entry_count > 0:
            # create a queue to write errors to
            error_queue = Queue.Queue()

            # create db writer
            writer = LmdbWriter(
                dataset_dir,
                stage,
                total_records=entry_count,
                feature_encoding=feature_encoding,
                label_encoding=label_encoding)
            writer.daemon = True
            writer.start()

            # create and fill encoder queue
            encoder_queue = Queue.Queue()
            batch = []
            for entry_id in entry_ids:
                batch.append(entry_id)
                if len(batch) >= batch_size:
                    # queue this batch
                    encoder_queue.put(batch)
                    batch = []
            if len(batch) > 0:
                # queue any remaining entries
                encoder_queue.put(batch)

            # create encoder threads
            encoders = []
            for _ in xrange(num_threads):
                encoder = Encoder(encoder_queue, writer, extension, error_queue)
                encoder.daemon = True
                encoder.start()
                encoders.append(encoder)

            # wait for all encoder threads to complete and aggregate data
            feature_sum = None
            processed_count = 0
            feature_shape = None
            label_shape = None
            for encoder in encoders:
                encoder.join()
                if feature_sum is None:
                    feature_sum = encoder.feature_sum
                elif encoder.feature_sum is not None:
                    feature_sum += encoder.feature_sum
                if feature_shape is None:
                    feature_shape = encoder.feature_shape
                    logger.info('Feature shape for stage %s: %s' % (stage, repr(feature_shape)))
                elif encoder.feature_shape is not None:
                    assert feature_shape == encoder.feature_shape
                if label_shape is None:
                    label_shape = encoder.label_shape
                    logger.info('Label shape for stage %s: %s' % (stage, repr(label_shape)))
                elif encoder.label_shape is not None:
                    assert label_shape == encoder.label_shape
                processed_count += encoder.processed_count

            # write mean file
            if feature_sum is not None:
                self.save_mean(feature_sum, processed_count, dataset_dir, stage)

            # wait for writer thread to complete
            writer.set_done()
            writer.join()

            # catch errors that may have occurred in reader threads
            if not error_queue.empty():
                while not error_queue.empty():
                    err = error_queue.get()
                    logger.error(err)
                raise Exception(err)

            if processed_count != entry_count:
                # TODO: handle this more gracefully
                raise ValueError('Number of processed entries (%d) does not match entry count (%d)' % (processed_count, entry_count))

            logger.info('Found %d entries for stage %s' % (processed_count, stage))


    def save_mean(self, feature_sum, entry_count, dataset_dir, stage):
        """
        Save mean to file
        """
        data = np.around(feature_sum / entry_count).astype(np.uint8)
        mean_file = os.path.join(stage, 'mean.binaryproto')
        # Transform to caffe's format requirements
        if data.ndim == 3:
            if data.shape[0] == 3:
                # channel swap
                # XXX see issue #59
                data = data[[2, 1, 0], ...]
        elif data.ndim == 2:
            # Add a channels axis
            data = data[np.newaxis, :, :]

        blob = caffe_pb2.BlobProto()
        blob.num = 1
        blob.channels, blob.height, blob.width = data.shape
        blob.data.extend(data.astype(float).flat)

        with open(os.path.join(dataset_dir, mean_file), 'wb') as outfile:
            outfile.write(blob.SerializeToString())

        logger.info('Created mean file for stage %s in %s' % (stage, mean_file))


def create_generic_db(jobs_dir, dataset_id, stage):
    """
    Create a generic DB
    """

    # job directory defaults to that defined in DIGITS config
    if jobs_dir == 'none':
        jobs_dir = digits.config.config_value('jobs_dir')

    # load dataset job
    dataset_dir = os.path.join(jobs_dir, dataset_id)
    assert os.path.isdir(dataset_dir), "Dataset dir %s does not exist" % dataset_dir
    dataset = Job.load(dataset_dir)

    # create instance of extension
    extension_id = dataset.extension_id
    extension_class = extensions.data.get_extension(extension_id)
    extension = extension_class(**dataset.extension_userdata)

    # encoding
    feature_encoding = dataset.feature_encoding
    label_encoding = dataset.label_encoding

    batch_size = dataset.batch_size
    num_threads = dataset.num_threads

    # create main DB creator object and execute main method
    db_creator = DbCreator()
    db_creator.create_db(
        extension,
        stage,
        dataset_dir,
        batch_size,
        num_threads,
        feature_encoding,
        label_encoding)

    logger.info('Generic DB creation Done')

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='DB creation tool - DIGITS')

    ### Positional arguments

    parser.add_argument(
        'dataset',
        help='Dataset Job ID')

    ### Optional arguments
    parser.add_argument(
        '-j',
        '--jobs_dir',
        default='none',
        help='Jobs directory (default: from DIGITS config)',
        )

    parser.add_argument(
        '-s',
        '--stage',
        default='train',
        help='Stage (train, val, test)',
        )

    args = vars(parser.parse_args())

    try:
        create_generic_db(
            args['jobs_dir'],
            args['dataset'],
            args['stage']
            )
    except Exception as e:
        logger.error('%s: %s' % (type(e).__name__, e.message))
        raise
