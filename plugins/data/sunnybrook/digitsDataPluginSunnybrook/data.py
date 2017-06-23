# Copyright (c) 2016-2017, NVIDIA CORPORATION.  All rights reserved.
from __future__ import absolute_import

import cv2
import fnmatch
import math
import os
import random
import re

import dicom
import numpy as np

from digits.utils import subclass, override, constants
from digits.utils.constants import COLOR_PALETTE_ATTRIBUTE
from digits.extensions.data.interface import DataIngestionInterface
from .forms import DatasetForm, InferenceForm


DATASET_TEMPLATE = "templates/dataset_template.html"
INFERENCE_TEMPLATE = "templates/inference_template.html"


# This is the subset of SAX series to use for Left Ventricle segmentation
# in the challenge training dataset
SAX_SERIES = {

    "SC-HF-I-1": "0004",
    "SC-HF-I-2": "0106",
    "SC-HF-I-4": "0116",
    "SC-HF-I-40": "0134",
    "SC-HF-NI-3": "0379",
    "SC-HF-NI-4": "0501",
    "SC-HF-NI-34": "0446",
    "SC-HF-NI-36": "0474",
    "SC-HYP-1": "0550",
    "SC-HYP-3": "0650",
    "SC-HYP-38": "0734",
    "SC-HYP-40": "0755",
    "SC-N-2": "0898",
    "SC-N-3": "0915",
    "SC-N-40": "0944",
}

#
# Utility functions
#


def shrink_case(case):
    toks = case.split("-")

    def shrink_if_number(x):
        try:
            cvt = int(x)
            return str(cvt)
        except ValueError:
            return x
    return "-".join([shrink_if_number(t) for t in toks])


class Contour(object):

    def __init__(self, ctr_path):
        self.ctr_path = ctr_path
        match = re.search(r"/([^/]*)/contours-manual/IRCCI-expert/IM-0001-(\d{4})-icontour-manual.txt", ctr_path)
        self.case = shrink_case(match.group(1))
        self.img_no = int(match.group(2))

    def __str__(self):
        return "<Contour for case %s, image %d>" % (self.case, self.img_no)

    __repr__ = __str__


def get_all_contours(contour_path):
    # walk the directory structure for all the contour files
    contours = [
        os.path.join(dirpath, f)
        for dirpath, dirnames, files in os.walk(contour_path)
        for f in fnmatch.filter(files, 'IM-0001-*-icontour-manual.txt')
    ]
    extracted = map(Contour, contours)
    return extracted


def load_contour(contour, img_path):
    filename = "IM-%s-%04d.dcm" % (SAX_SERIES[contour.case], contour.img_no)
    full_path = os.path.join(img_path, contour.case, filename)
    img = load_image(full_path)
    ctrs = np.loadtxt(contour.ctr_path, delimiter=" ").astype(np.int)
    label = np.zeros_like(img, dtype="uint8")
    cv2.fillPoly(label, [ctrs], 1)
    return img, label


def load_image(full_path):
    f = dicom.read_file(full_path)
    return f.pixel_array.astype(np.int)


@subclass
class DataIngestion(DataIngestionInterface):
    """
    A data ingestion extension for the Sunnybrook dataset
    """

    def __init__(self, is_inference_db=False, **kwargs):
        super(DataIngestion, self).__init__(**kwargs)

        self.userdata['is_inference_db'] = is_inference_db

        self.userdata['class_labels'] = ['background', 'left ventricle']

        # get list of contours
        if 'contours' not in self.userdata:
            contours = get_all_contours(self.contour_folder)
            random.shuffle(contours)
            self.userdata['contours'] = contours
        else:
            contours = self.userdata['contours']

        # get number of validation entries
        pct_val = int(self.folder_pct_val)
        self.userdata['n_val_entries'] = int(math.floor(len(contours) * pct_val / 100))

        # label palette (0->black (background), 1->white (foreground), others->black)
        palette = [0, 0, 0,  255, 255, 255] + [0] * (254 * 3)
        self.userdata[COLOR_PALETTE_ATTRIBUTE] = palette

    @override
    def encode_entry(self, entry):
        if isinstance(entry, basestring):
            img = load_image(entry)
            label = np.array([0])
        else:
            img, label = load_contour(entry, self.image_folder)
            label = label[np.newaxis, ...]

        if self.userdata['channel_conversion'] == 'L':
            feature = img[np.newaxis, ...]
        elif self.userdata['channel_conversion'] == 'RGB':
            feature = np.empty(shape=(3, img.shape[0], img.shape[1]), dtype=img.dtype)
            # just copy the same data over the three color channels
            feature[0] = img
            feature[1] = img
            feature[2] = img

        return feature, label

    @staticmethod
    @override
    def get_category():
        return "Images"

    @staticmethod
    @override
    def get_id():
        return "image-sunnybrook"

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
        n_val_entries = self.userdata['n_val_entries']
        form = InferenceForm()
        for idx, ctr in enumerate(self.userdata['contours'][:n_val_entries]):
            form.validation_record.choices.append((str(idx), ctr.case))
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
        return "Sunnybrook LV Segmentation"

    @override
    def itemize_entries(self, stage):
        ctrs = self.userdata['contours']
        n_val_entries = self.userdata['n_val_entries']

        entries = []
        if not self.userdata['is_inference_db']:
            if stage == constants.TRAIN_DB:
                entries = ctrs[n_val_entries:]
            elif stage == constants.VAL_DB:
                entries = ctrs[:n_val_entries]
        elif stage == constants.TEST_DB:
            if self.userdata['validation_record'] != 'none':
                if self.userdata['test_image_file']:
                    raise ValueError("Specify either an image or a record "
                                     "from the validation set.")
                # test record from validation set
                entries = [ctrs[int(self.validation_record)]]

            else:
                # test image file
                entries = [self.userdata['test_image_file']]

        return entries
