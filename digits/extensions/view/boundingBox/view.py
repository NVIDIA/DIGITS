# Copyright (c) 2016-2017, NVIDIA CORPORATION.  All rights reserved.
from __future__ import absolute_import

import os
import PIL.Image

import digits
from digits.utils import subclass, override
from .forms import ConfigForm
from ..interface import VisualizationInterface

CONFIG_TEMPLATE = "config_template.html"
HEADER_TEMPLATE = "header_template.html"
APP_BEGIN_TEMPLATE = "app_begin_template.html"
APP_END_TEMPLATE = "app_end_template.html"
VIEW_TEMPLATE = "view_template.html"


@subclass
class Visualization(VisualizationInterface):
    """
    A visualization extension to display bounding boxes
    """

    def __init__(self, dataset, **kwargs):
        # memorize view template for later use
        extension_dir = os.path.dirname(os.path.abspath(__file__))
        self.view_template = open(
            os.path.join(extension_dir, VIEW_TEMPLATE), "r").read()

        # stats
        self.image_count = 0
        self.bbox_count = 0

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
        context = {'form': form}
        return (template, context)

    @override
    def get_header_template(self):
        """
        Implements get_header_template() method from view extension interface
        """
        extension_dir = os.path.dirname(os.path.abspath(__file__))
        template = open(
            os.path.join(extension_dir, HEADER_TEMPLATE), "r").read()
        return template, {'image_count': self.image_count, 'bbox_count': self.bbox_count}

    @override
    def get_ng_templates(self):
        """
        Implements get_ng_templates() method from view extension interface
        """
        extension_dir = os.path.dirname(os.path.abspath(__file__))
        header = open(os.path.join(extension_dir, APP_BEGIN_TEMPLATE), "r").read()
        footer = open(os.path.join(extension_dir, APP_END_TEMPLATE), "r").read()
        return header, footer

    @staticmethod
    def get_id():
        return 'image-bounding-boxes'

    @staticmethod
    def get_title():
        return 'Bounding boxes'

    @override
    def get_view_template(self, data):
        """
        return:
        - (template, context) tuple
          - template is a Jinja template to use for rendering config options
          - context is a dictionary of context variables to use for rendering
          the form
        """
        return self.view_template, {
            'image': data['image'],
            'bboxes': data['bboxes'],
            'index': data['index'],
        }

    @override
    def process_data(
            self,
            input_id,
            input_data,
            inference_data,
            ground_truth=None):
        """
        Process one inference output
        """
        # get source image
        image = PIL.Image.fromarray(input_data).convert('RGB')

        self.image_count += 1

        # create arrays in expected format
        keys = inference_data.keys()
        bboxes = dict(zip(keys, [[] for x in range(0, len(keys))]))
        for key, outputs in inference_data.items():
            # last number is confidence
            bboxes[key] = [list(o) for o in outputs if o[-1] > 0]
            self.bbox_count += len(bboxes[key])
        image_html = digits.utils.image.embed_image_html(image)

        return {
            'image': image_html,
            'bboxes': bboxes,
            'index': self.image_count,
        }
