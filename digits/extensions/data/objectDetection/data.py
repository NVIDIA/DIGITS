# Copyright (c) 2016-2017, NVIDIA CORPORATION.  All rights reserved.
from __future__ import absolute_import

import csv
import operator
import os
import random
import StringIO

import numpy as np

import digits
from digits.utils import subclass, override, constants
from ..interface import DataIngestionInterface
from .forms import DatasetForm
from .utils import GroundTruth, GroundTruthObj
from .utils import bbox_to_array, pad_image, resize_bbox_list

TEMPLATE = "template.html"


@subclass
class DataIngestion(DataIngestionInterface):
    """
    A data ingestion extension for an object detection dataset
    """

    def __init__(self, **kwargs):
        super(DataIngestion, self).__init__(**kwargs)

        # this instance is automatically populated with form field
        # attributes by superclass constructor

        if hasattr(self, 'custom_classes') and self.custom_classes != '':
            s = StringIO.StringIO(self.custom_classes)
            reader = csv.reader(s)
            self.class_mappings = {}
            for idx, name in enumerate(reader.next()):
                self.class_mappings[name.strip().lower()] = idx
        else:
            self.class_mappings = None

        # this will be set when we know the phase we are encoding
        self.ground_truth = None

    @override
    def encode_entry(self, entry):
        """
        Return numpy.ndarray
        """
        image_filename = entry

        # (1) image part

        # load from file (this returns a PIL image)
        img = digits.utils.image.load_image(image_filename)
        if self.channel_conversion != 'none':
            if img.mode != self.channel_conversion:
                # convert to different image mode if necessary
                img = img.convert(self.channel_conversion)

        # note: the form validator ensured that either none
        # or both width/height were specified
        if self.padding_image_width:
            # pad image
            img = pad_image(
                img,
                self.padding_image_height,
                self.padding_image_width)

        if self.resize_image_width is not None:
            resize_ratio_x = float(self.resize_image_width) / img.size[0]
            resize_ratio_y = float(self.resize_image_height) / img.size[1]
            # resize and convert to numpy HWC
            img = digits.utils.image.resize_image(
                img,
                self.resize_image_height,
                self.resize_image_width)
        else:
            resize_ratio_x = 1
            resize_ratio_y = 1
            # convert to numpy array
            img = np.array(img)

        if img.ndim == 2:
            # grayscale
            img = img[np.newaxis, :, :]
            if img.dtype == 'uint16':
                img = img.astype(float)
        else:
            if img.ndim != 3 or img.shape[2] != 3:
                raise ValueError("Unsupported image shape: %s" % repr(img.shape))
            # HWC -> CHW
            img = img.transpose(2, 0, 1)

        # (2) label part

        # make sure label exists
        label_id = os.path.splitext(os.path.basename(entry))[0]

        if label_id not in self.datasrc_annotation_dict:
            raise ValueError("Label key %s not found in label folder" % label_id)
        annotations = self.datasrc_annotation_dict[label_id]

        # collect bbox list into bboxList
        bboxList = []

        for bbox in annotations:
            # retrieve all vars defining groundtruth, and interpret all
            # serialized values as float32s:
            np_bbox = np.array(bbox.gt_to_lmdb_format())
            bboxList.append(np_bbox)

        bboxList = sorted(
            bboxList,
            key=operator.itemgetter(GroundTruthObj.lmdb_format_length() - 1)
        )

        bboxList.reverse()

        # adjust bboxes according to image cropping
        bboxList = resize_bbox_list(bboxList, resize_ratio_x, resize_ratio_y)

        # return data
        feature = img
        label = np.asarray(bboxList)

        # LMDB compaction: now label (aka bbox) is the joint array
        label = bbox_to_array(
            label,
            0,
            max_bboxes=self.max_bboxes,
            bbox_width=GroundTruthObj.lmdb_format_length())

        return feature, label

    @staticmethod
    @override
    def get_category():
        return "Images"

    @staticmethod
    @override
    def get_id():
        return "image-object-detection"

    @staticmethod
    @override
    def get_dataset_form():
        return DatasetForm()

    @staticmethod
    @override
    def get_dataset_template(form):
        """
        parameters:
        - form: form returned by get_dataset_form(). This may be populated with values if the job was cloned
        return:
        - (template, context) tuple
          template is a Jinja template to use for rendering dataset creation options
          context is a dictionary of context variables to use for rendering the form
        """
        extension_dir = os.path.dirname(os.path.abspath(__file__))
        template = open(os.path.join(extension_dir, TEMPLATE), "r").read()
        context = {'form': form}
        return (template, context)

    @staticmethod
    @override
    def get_title():
        return "Object Detection"

    @override
    def itemize_entries(self, stage):
        """
        return list of image file names to encode for specified stage
        """
        if stage == constants.TEST_DB:
            # don't retun anything for the test stage
            return []
        elif stage == constants.TRAIN_DB:
            # load ground truth
            self.load_ground_truth(self.train_label_folder)
            # get training image file names
            return self.make_image_list(self.train_image_folder)
        elif stage == constants.VAL_DB:
            if self.val_image_folder != '':
                # load ground truth
                self.load_ground_truth(
                    self.val_label_folder,
                    self.val_min_box_size)
                # get validation image file names
                return self.make_image_list(self.val_image_folder)
            else:
                # no validation folder was specified
                return []
        else:
            raise ValueError("Unknown stage: %s" % stage)

    def load_ground_truth(self, folder, min_box_size=None):
        """
        load ground truth from specified folder
        """
        datasrc = GroundTruth(
            folder,
            min_box_size=min_box_size,
            class_mappings=self.class_mappings)
        datasrc.load_gt_obj()
        self.datasrc_annotation_dict = datasrc.objects_all

        scene_files = []
        for key in self.datasrc_annotation_dict:
            scene_files.append(key)

        # determine largest label height:
        self.max_bboxes = max([len(annotation) for annotation in self.datasrc_annotation_dict.values()])

    def make_image_list(self, folder):
        """
        find all supported images within specified folder and return list of file names
        """
        image_files = []
        for dirpath, dirnames, filenames in os.walk(folder, followlinks=True):
            for filename in filenames:
                if filename.lower().endswith(digits.utils.image.SUPPORTED_EXTENSIONS):
                    image_files.append('%s' % os.path.join(dirpath, filename))
        if len(image_files) == 0:
            raise ValueError("Unable to find supported images in %s" % folder)
        # shuffle
        random.shuffle(image_files)
        return image_files
