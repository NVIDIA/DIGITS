# Copyright (c) 2014-2015, NVIDIA CORPORATION.  All rights reserved.

from digits.utils import subclass, override
from digits.status import Status
from ..job import ImageDatasetJob
from digits.dataset import tasks

# NOTE: Increment this everytime the pickled object changes
PICKLE_VERSION = 2

@subclass
class ImageClassificationDatasetJob(ImageDatasetJob):
    """
    A Job that creates an image dataset for a classification network
    """

    def __init__(self, **kwargs):
        super(ImageClassificationDatasetJob, self).__init__(**kwargs)
        self.pickver_job_dataset_image_classification = PICKLE_VERSION

        self.labels_file = None

    def __setstate__(self, state):
        super(ImageClassificationDatasetJob, self).__setstate__(state)

        if self.pickver_job_dataset_image_classification <= 1:
            print 'Upgrading ImageClassificationDatasetJob to version 2'
            task = self.train_db_task()
            if task.image_dims[2] == 3:
                if task.encoding == "jpg":
                    if task.mean_file.endswith('.binaryproto'):
                        print '\tConverting mean file "%s" from RGB to BGR.' % task.path(task.mean_file)
                        import numpy as np
                        import caffe_pb2

                        old_blob = caffe_pb2.BlobProto()
                        with open(task.path(task.mean_file),'rb') as infile:
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
                        with open(task.path(task.mean_file), 'wb') as outfile:
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
        return 'Image Classification Dataset'

    @override
    def train_db_task(self):
        """
        Return the task that creates the training set
        """
        for t in self.tasks:
            if isinstance(t, tasks.CreateDbTask) and 'train' in t.name().lower():
                return t
        return None

