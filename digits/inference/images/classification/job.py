# Copyright (c) 2016, NVIDIA CORPORATION.  All rights reserved.
from __future__ import absolute_import

from ..job import ImageInferenceJob
from digits.job import Job
from digits.utils import subclass, override

@subclass
class ImageInferenceClassificationJob(ImageInferenceJob):
    """
    A Job that exercises the forward pass of a classification image neural network
    """

    @override
    def job_type(self):
        return 'Image Classification Inference'

@subclass
class ImageInferenceClassifyOneJob(ImageInferenceClassificationJob):
    """
    A Job that exercises the forward pass of a classification image neural network
    Inference exercised through 'classify_one' method
    """

    @override
    def job_type(self):
        return 'Image Classification Classify One Inference'

@subclass
class ImageInferenceClassifyManyJob(ImageInferenceClassificationJob):
    """
    A Job that exercises the forward pass of a classification image neural network
    Inference exercised through 'classify_many' method
    """

    @override
    def job_type(self):
        return 'Image Classification Classify Many Inference'

@subclass
class ImageInferenceTopNJob(ImageInferenceClassificationJob):
    """
    A Job that exercises the forward pass of a classification image neural network
    Inference exercised through 'top N' method
    """

    @override
    def __init__(self, **kwargs):
        """
        Keyword arguments:
        top_n -- number of images to show in TopN view
        """

        self.top_n = kwargs.pop('top_n', None)

        super(ImageInferenceTopNJob, self).__init__(**kwargs)

    @override
    def job_type(self):
        return 'Image Classification TopN Inference'

    @override
    def get_parameters(self):
        """Return inference parameters"""
        return super(ImageInferenceTopNJob, self).get_parameters() + (self.top_n,)
