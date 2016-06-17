# Copyright (c) 2016, NVIDIA CORPORATION.  All rights reserved.
from __future__ import absolute_import

import os
import PIL.Image

import digits
from digits.utils import subclass, override
from .forms import ConfigForm
from ..interface import VisualizationInterface

CONFIG_TEMPLATE = "config_template.html"
HEADER_TEMPLATE = "header_template.html"
VIEW_TEMPLATE = "view_template.html"


@subclass
class Visualization(VisualizationInterface):
    """
    A visualization extension to display bounding boxes
    """

    def __init__(self, dataset, **kwargs):
        # bounding box options
        color = kwargs['box_color']
        if color == "red":
            self.color = (255, 0, 0)
        elif color == "green":
            self.color = (0, 255, 0)
        elif color == "blue":
            self.color = (0, 0, 255)
        else:
            raise ValueError("unknown color: %s" % color)
        self.line_width = int(kwargs['line_width'])

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
        return self.view_template, {'image': data['image']}

    @override
    def process_data(
            self,
            dataset,
            input_data,
            inference_data,
            ground_truth=None):
        """
        Process one inference output
        Parameters:
        - dataset: dataset used during training
        - input_data: input to the network
        - inference_data: network output
        - ground_truth: Ground truth. Format is application specific.
          None if absent.
        Returns:
        - an object reprensenting the processed data
        """
        # get source image
        image = PIL.Image.fromarray(input_data).convert('RGB')

        self.image_count += 1

        # create arrays in expected format
        bboxes = []
        outputs = inference_data[inference_data.keys()[0]]
        for output in outputs:
            # last number is confidence
            if output[-1] > 0:
                box = ((output[0], output[1]), (output[2], output[3]))
                bboxes.append(box)
                self.bbox_count += 1
        digits.utils.image.add_bboxes_to_image(
            image,
            bboxes,
            self.color,
            self.line_width)
        image_html = digits.utils.image.embed_image_html(image)
        return {'image': image_html}
