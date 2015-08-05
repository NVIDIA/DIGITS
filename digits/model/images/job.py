# Copyright (c) 2014-2015, NVIDIA CORPORATION.  All rights reserved.

from ..job import ModelJob, PretrainedModelJob

# NOTE: Increment this everytime the pickled object changes
PICKLE_VERSION = 1

class ImageModelJob(ModelJob):
    """
    A Job that creates an image model
    """

    def __init__(self, **kwargs):
        """
        """
        super(ImageModelJob, self).__init__(**kwargs)
        self.pickver_job_model_image = PICKLE_VERSION

class ImagePretrainedModelJob(PretrainedModelJob):
    """
    A Job that creates an image model for pretrained models.
    """

    def __init__(self, **kwargs):
        """
        """
        super(ImagePretrainedModelJob, self).__init__(**kwargs)
        self.pickver_job_pretrainedmodel_image = PICKLE_VERSION

