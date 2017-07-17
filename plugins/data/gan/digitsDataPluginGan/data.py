# Copyright (c) 2016, NVIDIA CORPORATION.  All rights reserved.
from __future__ import absolute_import

import os
import pickle

import numpy as np

from digits.utils import constants, override, image, subclass
from digits.extensions.data.interface import DataIngestionInterface
from .forms import DatasetForm, InferenceForm


DATASET_TEMPLATE = "templates/dataset_template.html"
INFERENCE_TEMPLATE = "templates/inference_template.html"

CELEBA_ALL_ATTRIBUTES = """
                         5_o_Clock_Shadow Arched_Eyebrows Attractive Bags_Under_Eyes Bald Bangs
                         Big_Lips Big_Nose Black_Hair Blond_Hair Blurry Brown_Hair Bushy_Eyebrows
                         Chubby Double_Chin Eyeglasses Goatee Gray_Hair Heavy_Makeup High_Cheekbones
                         Male Mouth_Slightly_Open Mustache Narrow_Eyes No_Beard Oval_Face Pale_Skin
                         Pointy_Nose Receding_Hairline Rosy_Cheeks Sideburns Smiling Straight_Hair
                         Wavy_Hair Wearing_Earrings Wearing_Hat Wearing_Lipstick Wearing_Necklace
                         Wearing_Necktie Young
                        """.split()

CELEBA_EDITABLE_ATTRIBUTES = [
    'Bald', 'Black_Hair', 'Blond_Hair', 'Male', 'Smiling', 'Wearing_Lipstick', 'Young'
]

CELEBA_EDITABLE_ATTRIBUTES_IDS = [CELEBA_ALL_ATTRIBUTES.index(attr) for attr in CELEBA_EDITABLE_ATTRIBUTES]


def one_hot(val, depth):
    x = np.zeros(depth)
    x[val] = 1
    return x


def slerp(val, low, high):
    """Spherical interpolation. val has a range of 0 to 1."""
    if val <= 0:
        return low
    elif val >= 1:
        return high
    omega = np.arccos(np.dot(low/np.linalg.norm(low), high/np.linalg.norm(high)))
    so = np.sin(omega)
    return np.sin((1.0-val)*omega) / so * low + np.sin(val*omega)/so * high


def parse_lines_of_floats(s):
    return [[float(val) for val in line.split()] for line in s.splitlines()]


@subclass
class DataIngestion(DataIngestionInterface):
    """
    A data ingestion extension for GANs
    """

    # CONFIG = "mnist"
    CONFIG = "celeba"
    # CONFIG = "celeba_cond"

    def __init__(self, is_inference_db=False, **kwargs):
        super(DataIngestion, self).__init__(**kwargs)

        if 'dataset_type' in self.userdata:
            self.CONFIG = self.userdata['dataset_type']

        self.z_dim = 100
        if self.CONFIG == "mnist":
            self.y_dim = 10
        elif self.CONFIG == "celeba":
            self.y_dim = 0
        elif self.CONFIG == "celeba_cond":
            self.y_dim = 40

        self.userdata['is_inference_db'] = is_inference_db

        self.input_dim = self.z_dim + self.y_dim

    @override
    def encode_entry(self, entry):
        if not self.userdata['is_inference_db']:
            filename = entry[0]
            label = entry[1]
            feature = self.scale_image(filename)
            label = np.array(label).reshape(1, 1, len(label))
        else:
            if self.userdata['task_id'] in ['style',
                                            'class',
                                            'genimg',
                                            'attributes',
                                            'analogy',
                                            'animation']:
                feature = entry
                label = np.array([0])
            elif self.userdata['task_id'] == 'enclist':
                filename = entry[0]
                label = entry[1]
                feature = self.scale_image(filename)
                label = np.array(label).reshape(1, 1, len(label))
            else:
                raise NotImplementedError
        return feature, label

    def encode_PIL_Image(self, image):
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
        return "image-gan"

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
        return InferenceForm(CELEBA_ALL_ATTRIBUTES, CELEBA_EDITABLE_ATTRIBUTES_IDS)

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
        return "GAN"

    @override
    def itemize_entries(self, stage):
        entries = []
        if not self.userdata['is_inference_db']:
            if stage == constants.TRAIN_DB:
                # read file list
                with open(self.userdata['file_list']) as f:
                    lines = f.read().splitlines()
                    # skip first 2 lines (header)
                    for line in lines[2:]:
                        fields = line.split()
                        filename = fields[0]
                        # add full path
                        filename = os.path.join(self.userdata['image_folder'], filename)
                        label = [int(field) for field in fields[1:]]
                        entries.append((filename, label))
        elif stage == constants.TEST_DB:
            if self.userdata['task_id'] == 'style':
                if self.userdata['style_z1_vector']:
                    z1 = np.array([float(v) for v in self.userdata['style_z1_vector'].split()])
                else:
                    z1 = np.random.normal(size=(100,))
                if self.userdata['style_z2_vector']:
                    z2 = np.array([float(v) for v in self.userdata['style_z2_vector'].split()])
                else:
                    z2 = np.random.normal(size=(100,))
                for val in np.linspace(0, 1, self.userdata['row_count']):
                    for c in range(10):
                        z_ = slerp(val, z1, z2)
                        feature = np.append(z_, one_hot(c, self.y_dim)).reshape((1, 1, self.input_dim))
                        entries.append(feature)
            elif self.userdata['task_id'] == 'class':
                if self.userdata['class_z_vector']:
                    z = np.array([float(v) for v in self.userdata['class_z_vector'].split()])
                else:
                    z = np.random.normal(size=(100,))
                for val in np.linspace(0, 1, self.userdata['row_count']):
                    for i in range(10):
                        c_0 = i
                        c_1 = (i + 1) % 10
                        feature_0 = np.append(z, one_hot(c_0, self.y_dim))
                        feature_1 = np.append(z, one_hot(c_1, self.y_dim))
                        feature = slerp(val, feature_0, feature_1).reshape((1, 1, self.input_dim))
                        entries.append(feature)
            elif self.userdata['task_id'] == 'genimg':
                c = int(self.userdata['genimg_class_id'])
                if self.userdata['genimg_z_vector']:
                    z = np.array([float(v) for v in self.userdata['genimg_z_vector'].split()])
                else:
                    z = np.random.normal(size=(100,))
                if self.y_dim > 0:
                    z = np.append(z, one_hot(c, self.y_dim))
                feature = z.reshape((1, 1, self.input_dim))
                entries.append(feature)
            elif self.userdata['task_id'] == 'attributes':
                if self.userdata['attributes_z_vector']:
                    z = np.array([float(v) for v in self.userdata['attributes_z_vector'].split()])
                else:
                    z = np.random.normal(size=(100,))
                with open(self.userdata['attributes_file'], 'rb') as f:
                    attributes_z = pickle.load(f)
                    params = parse_lines_of_floats(self.userdata['attributes_params'])
                    for img_params in params:
                        z_img = np.copy(z)
                        for i, coeff in enumerate(img_params):
                            z_img += coeff * attributes_z[CELEBA_EDITABLE_ATTRIBUTES_IDS[i]]
                        entries.append(z_img.reshape((1, 1, self.input_dim)))
            elif self.userdata['task_id'] == 'enclist':
                with open(self.userdata['enc_file_list']) as f:
                    lines = f.read().splitlines()
                    # skip first 2 lines (header)
                    max_images = self.userdata['enc_num_images']
                    for line in lines[2:max_images + 2]:
                        fields = line.split()
                        filename = fields[0]
                        # add full path
                        filename = os.path.join(self.userdata['enc_image_folder'], filename)
                        label = [int(field) for field in fields[1:]]
                        entries.append((filename, label))
            elif self.userdata['task_id'] == 'analogy':
                if self.userdata['attributes_z1_vector']:
                    z1 = np.array([float(v) for v in self.userdata['attributes_z1_vector'].split()])
                else:
                    z1 = np.random.normal(size=(100,))
                if self.userdata['attributes_z2_vector']:
                    z2 = np.array([float(v) for v in self.userdata['attributes_z2_vector'].split()])
                else:
                    z2 = np.random.normal(size=(100,))
                if self.userdata['attributes_z3_vector']:
                    z3 = np.array([float(v) for v in self.userdata['attributes_z3_vector'].split()])
                else:
                    z3 = np.random.normal(size=(100,))

                # create analogy vector
                z4 = z2 + z3 - z1

                grid_size = self.userdata['row_count']

                # now interpolate across columns
                for row in xrange(grid_size):
                    row_k = row / float(grid_size - 1)
                    z_left = slerp(row_k, z1, z3)
                    z_right = slerp(row_k, z2, z4)
                    entries.append(z_left.reshape((1, 1, self.input_dim)))
                    for col in xrange(1, grid_size - 1):
                        col_k = col / float(grid_size - 1)
                        z = slerp(col_k, z_left, z_right)
                        entries.append(z.reshape((1, 1, self.input_dim)))
                    entries.append(z_right.reshape((1, 1, self.input_dim)))
            elif self.userdata['task_id'] == 'animation':
                zs = parse_lines_of_floats(self.userdata['animation_z_vectors'])
                zs = [np.array(z) for z in zs]
                num_transitions = self.userdata['animation_num_transitions']
                for i, z in enumerate(zs):
                    z_next = zs[(i + 1) % len(zs)]
                    for k in xrange(num_transitions):
                        z_ = slerp(float(k) / num_transitions, z, z_next)
                        entries.append(z_.reshape((1, 1, self.input_dim)))
            else:
                raise ValueError("Unknown task: %s" % self.userdata['task_id'])
        return entries

    def scale_image(self, filename):
        im = np.array(image.load_image(filename))

        # center crop
        if self.userdata['center_crop_size']:
            crop_size = int(self.userdata['center_crop_size'])
            width, height = im.shape[0:2]
            i = (width // 2) - crop_size // 2
            j = (height // 2) - crop_size // 2
            im = im[i:i + crop_size, j:j + crop_size, :]

        # resize
        if self.userdata['resize']:
            resize = int(self.userdata['resize'])
            im = image.resize_image(im, resize, resize, resize_mode='squash')

        # transpose to CHW
        feature = im.transpose(2, 0, 1)

        return feature
