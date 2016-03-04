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
class ImageInferenceClassifyManyJob(ImageInferenceClassificationJob):
    """
    A Job that exercises the forward pass of a classification image neural network
    Inference exercised through 'classify_many' method
    """

    @override
    def job_type(self):
        return 'Image Classification Classify Many Inference'