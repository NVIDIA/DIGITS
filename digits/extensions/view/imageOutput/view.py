# Copyright (c) 2016-2017, NVIDIA CORPORATION.  All rights reserved.
from __future__ import absolute_import

import os

import PIL.Image
import PIL.ImageDraw

import digits
from digits.utils import subclass, override
from .forms import ConfigForm
from ..interface import VisualizationInterface

CONFIG_TEMPLATE = "config_template.html"
VIEW_TEMPLATE = "view_template.html"


@subclass
class Visualization(VisualizationInterface):
    """
    A visualization extension to display the network output as an image
    """

    def __init__(self, dataset, **kwargs):
        # memorize view template for later use
        extension_dir = os.path.dirname(os.path.abspath(__file__))
        self.view_template = open(
            os.path.join(extension_dir, VIEW_TEMPLATE), "r").read()

        # view options
        self.channel_order = kwargs['channel_order'].upper()
        self.normalize = (kwargs['pixel_conversion'] == 'normalize')

    @staticmethod
    def get_config_form():
        return ConfigForm()

    @staticmethod
    def get_config_template(form):
        """
        parameters:
        - form: form returned by get_config_form(). This may be populated
        with values if the job was cloned
        returns:
        - (template, context) tuple
          - template is a Jinja template to use for rendering config options
          - context is a dictionary of context variables to use for rendering
          the form
        """
        extension_dir = os.path.dirname(os.path.abspath(__file__))
        template = open(
            os.path.join(extension_dir, CONFIG_TEMPLATE), "r").read()
        return (template, {'form': form})

    @staticmethod
    def get_id():
        return 'image-image-output'

    @staticmethod
    def get_title():
        return 'Image output'

    @override
    def get_view_template(self, data):
        """
        returns:
        - (template, context) tuple
          - template is a Jinja template to use for rendering config options
          - context is a dictionary of context variables to use for rendering
          the form
        """
        return self.view_template, {'image': digits.utils.image.embed_image_html(data)}

    @override
    def process_data(self, input_id, input_data, output_data):
        """
        Process one inference and return data to visualize
        """
        # assume the only output is a CHW image
        data = output_data[output_data.keys()[0]].astype('float32')
        channels = data.shape[0]
        if channels == 3 and self.channel_order == 'BGR':
            data = data[[2, 1, 0], ...]  # BGR to RGB
        # convert to HWC
        data = data.transpose((1, 2, 0))
        # assume 8-bit
        if self.normalize:
            data -= data.min()
            if data.max() > 0:
                data /= data.max()
                data *= 255
        else:
            # clip
            data = data.clip(0, 255)
        # convert to uint8
        data = data.astype('uint8')
        # convert to PIL image
        if channels == 1:
            # drop channel axis
            image = PIL.Image.fromarray(data[:, :, 0])
        elif channels == 3:
            image = PIL.Image.fromarray(data)
        else:
            raise ValueError("Unhandled number of channels: %d" % channels)

        return image
