# Copyright (c) 2014-2015, NVIDIA CORPORATION.  All rights reserved.

from ..job import DatasetJob

# NOTE: Increment this everytime the pickled object changes
PICKLE_VERSION = 1

class ImageDatasetJob(DatasetJob):
    """
    A Job that creates an image dataset
    """

    def __init__(self, image_dims, resize_mode, **kwargs):
        """
        Arguments:
        image_dims -- (height, width, channels)
        resize_mode -- used in utils.image.resize_image()
        """
        super(ImageDatasetJob, self).__init__(**kwargs)
        self.pickver_job_dataset_image = PICKLE_VERSION

        self.image_dims = image_dims
        self.resize_mode = resize_mode

