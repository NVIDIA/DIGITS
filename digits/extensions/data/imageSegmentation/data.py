# Copyright (c) 2016, NVIDIA CORPORATION.  All rights reserved.
from __future__ import absolute_import

# Find the best implementation available
try:
    from cStringIO import StringIO
except ImportError:
    from StringIO import StringIO
from collections import OrderedDict
import math
import os
import random

import caffe_pb2
import numpy as np
import PIL.Image
import PIL.ImagePalette

from digits.utils import image, subclass, override, constants
from digits.utils.constants import COLOR_PALETTE_ATTRIBUTE
from digits.utils.lmdbreader import DbReader
from ..interface import DataIngestionInterface
from .forms import DatasetForm

TEMPLATE = "template.html"


@subclass
class DataIngestion(DataIngestionInterface):
    """
    A data ingestion extension for an image segmentation dataset
    """

    def __init__(self, **kwargs):
        """
        the parent __init__ method automatically populates this
        instance with attributes from the form
        """
        super(DataIngestion, self).__init__(**kwargs)

        self.random_indices = None

        if 'seed' not in self.userdata:
            # choose random seed and add to userdata so it gets persisted
            self.userdata['seed'] = random.randint(0, 1000)

        if 'colormap_method' not in self.userdata:
            self.userdata['colormap_method'] = 'label'

        random.seed(self.userdata['seed'])

        if self.userdata['colormap_method'] == "label":
            # open first image in label folder to retrieve palette
            # all label images must use the same palette - this is enforced
            # during dataset creation
            filename = self.make_image_list(self.label_folder)[0]
            image = self.load_label(filename)
            self.userdata[COLOR_PALETTE_ATTRIBUTE] = image.getpalette()
        else:
            # read colormap from file
            with open(self.colormap_text_file) as f:
                palette = []
                lines = f.read().splitlines()
                for line in lines:
                    for val in line.split():
                        palette.append(int(val))
                # fill rest with zeros
                palette = palette + [0] * (256 * 3 - len(palette))
                self.userdata[COLOR_PALETTE_ATTRIBUTE] = palette
                self.palette_img = PIL.Image.new("P", (1, 1))
                self.palette_img.putpalette(palette)

        # get labels if those were provided
        if self.class_labels_file:
            with open(self.class_labels_file) as f:
                self.userdata['class_labels'] = f.read().splitlines()

    @override
    def encode_entry(self, entry):
        """
        Return numpy.ndarray
        """
        feature_image_file = entry[0]
        label_image_file = entry[1]

        # feature image
        feature_image = self.encode_PIL_Image(
            image.load_image(feature_image_file),
            self.channel_conversion)

        # label image
        label_image = self.load_label(label_image_file)
        if label_image.getpalette() != self.userdata[COLOR_PALETTE_ATTRIBUTE]:
            raise ValueError("All label images must use the same palette")
        label_image = self.encode_PIL_Image(label_image)

        return feature_image, label_image

    def encode_PIL_Image(self, image, channel_conversion='none'):
        if channel_conversion != 'none':
            if image.mode != channel_conversion:
                # convert to different image mode if necessary
                image = image.convert(channel_conversion)
        # convert to numpy array
        image = np.array(image)
        # add channel axis if input is grayscale image
        if image.ndim == 2:
            image = image[..., np.newaxis]
        elif image.ndim != 3:
            raise ValueError("Unhandled number of channels: %d" % image.ndim)
        # transpose to CHW
        image = image.transpose(2, 0, 1)
        return image

    @staticmethod
    @override
    def get_category():
        return "Images"

    @staticmethod
    @override
    def get_id():
        return "image-segmentation"

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
        returns:
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

    @staticmethod
    @override
    def get_title():
        return "Segmentation"

    @override
    def itemize_entries(self, stage):
        if stage == constants.TEST_DB:
            # don't retun anything for the test stage
            return []

        if stage == constants.TRAIN_DB or (not self.has_val_folder):
            feature_image_list = self.make_image_list(self.feature_folder)
            label_image_list = self.make_image_list(self.label_folder)
        else:
            # separate validation images
            feature_image_list = self.make_image_list(self.validation_feature_folder)
            label_image_list = self.make_image_list(self.validation_label_folder)

        # make sure filenames match
        if len(feature_image_list) != len(label_image_list):
            raise ValueError(
                "Expect same number of images in feature and label folders (%d!=%d)"
                % (len(feature_image_list), len(label_image_list)))

        for idx in range(len(feature_image_list)):
            feature_name = os.path.splitext(
                os.path.split(feature_image_list[idx])[1])[0]
            label_name = os.path.splitext(
                os.path.split(label_image_list[idx])[1])[0]
            if feature_name != label_name:
                raise ValueError("No corresponding feature/label pair found for (%s,%s)"
                                 % (feature_name, label_name))

        # split lists if there is no val folder
        if not self.has_val_folder:
            feature_image_list = self.split_image_list(feature_image_list, stage)
            label_image_list = self.split_image_list(label_image_list, stage)

        return zip(
            feature_image_list,
            label_image_list)

    def load_label(self, filename):
        """
        Load a label image
        """
        image = PIL.Image.open(filename)
        if self.userdata['colormap_method'] == "label":
            if image.mode not in ['P', 'L', '1']:
                raise ValueError("Labels are expected to be single-channel (paletted or "
                                 " grayscale) images - %s mode is '%s'"
                                 % (filename, image.mode))
        else:
            if image.mode not in ['RGB']:
                raise ValueError("Labels are expected to be RGB images - %s mode is '%s'"
                                 % (filename, image.mode))
            image = image.quantize(palette=self.palette_img)

        return image

    def make_image_list(self, folder):
        image_files = []
        for dirpath, dirnames, filenames in os.walk(folder, followlinks=True):
            for filename in filenames:
                if filename.lower().endswith(image.SUPPORTED_EXTENSIONS):
                    image_files.append('%s' % os.path.join(folder, filename))
        if len(image_files) == 0:
            raise ValueError("Unable to find supported images in %s" % folder)
        return sorted(image_files)

    def split_image_list(self, filelist, stage):
        if self.random_indices is None:
            self.random_indices = range(len(filelist))
            random.shuffle(self.random_indices)
        elif len(filelist) != len(self.random_indices):
            raise ValueError(
                "Expect same number of images in folders (%d!=%d)"
                % (len(filelist), len(self.random_indices)))
        filelist = [filelist[idx] for idx in self.random_indices]
        pct_val = int(self.folder_pct_val)
        n_val_entries = int(math.floor(len(filelist) * pct_val / 100))
        if stage == constants.VAL_DB:
            return filelist[:n_val_entries]
        elif stage == constants.TRAIN_DB:
            return filelist[n_val_entries:]
        else:
            raise ValueError("Unknown stage: %s" % stage)

    @staticmethod
    @override
    def can_explore():
        return True

    @staticmethod
    def get_data_from_db(db_path, page, size):
        reader = DbReader(db_path)
        data = []
        count = 0
        for key, value in reader.entries():
            if count >= page * size:
                datum = caffe_pb2.Datum()
                datum.ParseFromString(value)
                if not datum.encoded:
                    raise RuntimeError("Expected encoded database")
                s = StringIO()
                s.write(datum.data)
                s.seek(0)
                datum = PIL.Image.open(s)
                datum = np.array(datum)
                data.append(datum)
            count += 1
            if len(data) >= size:
                break
        return data, reader.total_entries

    @staticmethod
    def convert(data):
        """
        Convert the labael data to look like inference data
        :param data: label data
        :return: output
        - output is the data converted to look like inference data
        """
        idxs = np.unique(data)
        mx = np.max(idxs[np.where(idxs != 255)])
        output = [np.equal(data, i).astype('float') for i in range(mx + 1)]
        output = np.array(output)
        return output

    @staticmethod
    @override
    def get_data(feature_db_path, label_db_path, page, size):
        """
        Retrieve data from databases and return it as if it were inference data
        so that view extensions can present it in the Explore DB pages.
        :param feature_db_path: feature database path
        :param label_db_path: label database path
        :param page: which page
        :param size: number of items per page
        :return: (inputs, outputs, total_entries)
        - inputs are the feature data
        - outputs are the label data made to look like inference data
        - total_entries is the number of entries in the feature database
        """
        features, total_entries = DataIngestion.get_data_from_db(feature_db_path, page, size)
        inputs = {'data': features, 'ids': range(len(features))}

        labels, _ = DataIngestion.get_data_from_db(label_db_path, page, size)
        labels = [DataIngestion.convert(label) for label in labels]
        labels = np.array(labels)
        outputs = OrderedDict([('score', labels)])

        return inputs, outputs, total_entries
