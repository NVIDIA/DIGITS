# Copyright (c) 2014-2015, NVIDIA CORPORATION.  All rights reserved.

import os.path

from digits.dataset import tasks
from digits import utils
from digits.utils import subclass, override
from digits.status import Status
from ..job import ImageDatasetJob

# NOTE: Increment this everytime the pickled object changes
PICKLE_VERSION = 2

@subclass
class FeatureExtractionDatasetJob(ImageDatasetJob):
    """
    A Job that creates an image dataset for a feature extraction network
    """

    def __init__(self, **kwargs):
        super(FeatureExtractionDatasetJob, self).__init__(**kwargs)
        self.pickver_job_dataset_feature_extraction = PICKLE_VERSION

        self.labels_file = None

    def __setstate__(self, state):
        super(FeatureExtractionDatasetJob, self).__setstate__(state)

        if self.pickver_job_dataset_feature_extraction <= 1:
            print 'Upgrading FeatureExtractionDatasetJob to version 2'
            task = self.train_db_task()
            if task.image_dims[2] == 3:
                if task.encoding == "jpg":
                    if task.mean_file.endswith('.binaryproto'):
                        print '\tConverting mean file "%s" from RGB to BGR.' % task.path(task.mean_file)
                        try:
                            import caffe_pb2
                        except ImportError:
                            # See issue #32
                            from caffe.proto import caffe_pb2
                        import numpy as np

                        old_blob = caffe_pb2.BlobProto()
                        with open(task.path(task.mean_file)) as infile:
                            old_blob.ParseFromString(infile.read())
                        data = np.array(old_blob.data).reshape(
                                old_blob.channels,
                                old_blob.height,
                                old_blob.width)
                        data = data[[2,1,0],...] # channel swap
                        new_blob = caffe_pb2.BlobProto()
                        new_blob.num = 1
                        new_blob.channels, new_blob.height, new_blob.width = data.shape
                        new_blob.data.extend(data.astype(float).flat)
                        with open(task.path(task.mean_file), 'w') as outfile:
                            outfile.write(new_blob.SerializeToString())
                else:
                    print '\tSetting "%s" status to ERROR because it was created with RGB channels' % self.name()
                    self.status = Status.ERROR
                    for task in self.tasks:
                        task.status = Status.ERROR
                        task.exception = 'This dataset was created with unencoded RGB channels. Caffe requires BGR input.'

        self.pickver_job_dataset_image_classification = PICKLE_VERSION

    @override
    def job_type(self):
        return 'Feature Extraction Dataset'

    def from_files(self):
        """
        Checks for files already in the directory
        """
        assert len(self.tasks) == 0

        assert os.path.exists(self.path(utils.constants.TRAIN_FILE))
        assert os.path.exists(self.path(utils.constants.LABELS_FILE))
        self.labels_file = utils.constants.LABELS_FILE

        self.tasks.append(
                tasks.CreateDbTask(
                    job_dir     = self.dir(),
                    input_file  = utils.constants.TRAIN_FILE,
                    db_name     = utils.constants.TRAIN_DB,
                    image_dims  = self.image_dims,
                    resize_mode = self.resize_mode,
                    mean_file   = utils.constants.MEAN_FILE_CAFFE,
                    labels_file = self.labels_file,
                    )
                )
