# Copyright (c) 2016, NVIDIA CORPORATION.  All rights reserved.
from __future__ import absolute_import

import os
import re

import numpy as np

from digits.utils import subclass, override, constants
from digits.utils.constants import COLOR_PALETTE_ATTRIBUTE
from digits.extensions.data.interface import DataIngestionInterface
from .forms import DatasetForm, InferenceForm
from . import utils


DATASET_TEMPLATE = "templates/dataset_template.html"
INFERENCE_TEMPLATE = "templates/inference_template.html"


@subclass
class DataIngestion(DataIngestionInterface):
    """
    A data ingestion extension for the BRATS dataset
    """

    def __init__(self, is_inference_db=False, **kwargs):
        super(DataIngestion, self).__init__(**kwargs)

        self.userdata['is_inference_db'] = is_inference_db

        if 'files' not in self.userdata:
            files = utils.find_files(self.dataset_folder,
                                     self.group_id,
                                     self.modality)
            if not len(files):
                raise ValueError("Failed to find data files in %s for "
                                 "group %s and modality %s"
                                 % (self.dataset_folder, self.group_id, self.modality))
            self.userdata['files'] = files

        # label palette (0->black (background), 1->white (foreground), others->black)
        palette = [0, 0, 0,  255, 255, 255] + [0] * (254 * 3)
        self.userdata[COLOR_PALETTE_ATTRIBUTE] = palette

        self.userdata['class_labels'] = ['background', 'complete tumor']

    @override
    def encode_entry(self, entry):
        if self.userdata['is_inference_db']:
            # for inference, use image with maximum tumor area
            filter_method = 'max'
        else:
            filter_method = self.userdata['filter_method']
        feature, label = utils.encode_sample(entry, filter_method)

        data = []
        if feature.size > 0:
            if self.userdata['channel_conversion'] != 'none':
                # extract 2D slices: split across axial dimension
                features = np.split(feature, feature.shape[0])
                labels = np.split(label, label.shape[0])

                data = []
                for image, label in zip(features, labels):
                    if self.userdata['channel_conversion'] == 'L':
                        feature = image
                    elif self.userdata['channel_conversion'] == 'RGB':
                        image = image[0]
                        feature = np.empty(shape=(3, image.shape[0], image.shape[1]),
                                           dtype=image.dtype)
                        # just copy the same data over the three color channels
                        feature[:3] = [image, image, image]
                    data.append((feature, label))
            else:
                data.append((feature, label))
        return data

    @staticmethod
    @override
    def get_category():
        return "Images"

    @staticmethod
    @override
    def get_id():
        return "images-brats"

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
        template = open(os.path.join(extension_dir, DATASET_TEMPLATE), "r").read()
        context = {'form': form}
        return (template, context)

    @override
    def get_inference_form(self):
        all_entries = self.userdata['files']
        n_val_entries = int(len(all_entries)*self.userdata['pct_val']/100)
        val_entries = self.userdata['files'][:n_val_entries]
        form = InferenceForm()
        for idx, entry in enumerate(val_entries):
            match = re.match('.*pat(\d+)_.*', entry[0])
            if match:
                form.validation_record.choices.append((str(idx), 'Patient %s' % match.group(1)))
        return form

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
        return "Brain Tumor Segmentation"

    @override
    def itemize_entries(self, stage):
        all_entries = self.userdata['files']
        entries = []
        if not self.userdata['is_inference_db']:
            n_val_entries = int(len(all_entries)*self.pct_val/100)
            if stage == constants.TRAIN_DB:
                entries = all_entries[n_val_entries:]
            elif stage == constants.VAL_DB:
                entries = all_entries[:n_val_entries]
        elif stage == constants.TEST_DB:
            entries = [all_entries[int(self.validation_record)]]

        return entries
