# Copyright (c) 2016-2017, NVIDIA CORPORATION.  All rights reserved.
from __future__ import absolute_import

from digits.utils import subclass, override, constants
from digits.extensions.data.interface import DataIngestionInterface
from .forms import DatasetForm, InferenceForm

import numpy as np
import os

TEMPLATE = "templates/template.html"
INFERENCE_TEMPLATE = "templates/inference_template.html"


@subclass
class DataIngestion(DataIngestionInterface):
    """
    A data ingestion extension for an image gradient dataset
    """

    def __init__(self, is_inference_db=False, **kwargs):
        super(DataIngestion, self).__init__(**kwargs)

        self.userdata['is_inference_db'] = is_inference_db

        # Used to calculate the gradients later
        self.yy, self.xx = np.mgrid[:self.image_height,
                                    :self.image_width].astype('float')

    @override
    def encode_entry(self, entry):
        xslope, yslope = entry
        label = np.array([xslope, yslope])
        a = xslope * 255 / self.image_width
        b = yslope * 255 / self.image_height
        image = a * (self.xx - self.image_width / 2) + b * (self.yy - self.image_height / 2) + 127.5

        image = image.astype('uint8')

        # convert to 3D tensors
        image = image[np.newaxis, ...]
        label = label[np.newaxis, np.newaxis, ...]

        return image, label

    @staticmethod
    @override
    def get_category():
        return "Images"

    @staticmethod
    @override
    def get_id():
        return "image-gradients"

    @staticmethod
    @override
    def get_dataset_form():
        return DatasetForm()

    @staticmethod
    @override
    def get_dataset_template(form):
        """
        parameters:
        - form: form returned by get_dataset_form(). This may be populated
           with values if the job was cloned
        return:
        - (template, context) tuple
          - template is a Jinja template to use for rendering dataset creation
          options
          - context is a dictionary of context variables to use for rendering
          the form
        """
        extension_dir = os.path.dirname(os.path.abspath(__file__))
        template = open(os.path.join(extension_dir, TEMPLATE), "r").read()
        context = {'form': form}
        return (template, context)

    @override
    def get_inference_form(self):
        return InferenceForm()

    @staticmethod
    @override
    def get_inference_template(form):
        extension_dir = os.path.dirname(os.path.abspath(__file__))
        template = open(os.path.join(extension_dir, INFERENCE_TEMPLATE), "r").read()
        context = {'form': form}
        return (template, context)

    @staticmethod
    @override
    def get_title():
        return "Gradients"

    @override
    def itemize_entries(self, stage):
        count = 0
        if self.userdata['is_inference_db']:
            if stage == constants.TEST_DB:
                if self.test_image_count:
                    count = self.test_image_count
                else:
                    return [(self.gradient_x, self.gradient_y)]
        else:
            if stage == constants.TRAIN_DB:
                count = self.train_image_count
            elif stage == constants.VAL_DB:
                count = self.val_image_count
            elif stage == constants.TEST_DB:
                count = self.test_image_count
        return [np.random.random_sample(2) - 0.5 for i in xrange(count)] if count > 0 else []
