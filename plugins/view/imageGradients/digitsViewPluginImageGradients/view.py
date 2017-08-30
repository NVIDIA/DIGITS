# Copyright (c) 2016-2017, NVIDIA CORPORATION.  All rights reserved.
from __future__ import absolute_import

import numpy as np
import os
import PIL.Image
import PIL.ImageDraw

import digits
from digits.utils import subclass, override
from digits.extensions.view.interface import VisualizationInterface
from .forms import ConfigForm


CONFIG_TEMPLATE = "templates/config_template.html"
VIEW_TEMPLATE = "templates/view_template.html"


@subclass
class Visualization(VisualizationInterface):
    """
    A visualization extension to display image gradient magnitude and direction
    """

    def __init__(self, dataset, **kwargs):
        """
        Init
        """
        # arrow config
        arrow_color = kwargs['arrow_color']
        if arrow_color == "red":
            self.color = (255, 0, 0)
        elif arrow_color == "green":
            self.color = (0, 255, 0)
        elif arrow_color == "blue":
            self.color = (0, 0, 255)
        else:
            raise ValueError("unknown color: %s" % arrow_color)
        self.arrow_size = float(kwargs['arrow_size'])

        # image dimensions (HWC)
        image_shape = dataset.get_feature_dims()
        self.height = image_shape[0]
        self.width = image_shape[1]

        # memorize view template for later use
        extension_dir = os.path.dirname(os.path.abspath(__file__))
        self.view_template = open(
            os.path.join(extension_dir, VIEW_TEMPLATE), "r").read()

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

    @staticmethod
    def get_id():
        return "image-gradients"

    @staticmethod
    def get_title():
        return "Gradients"

    @override
    def get_view_template(self, data):
        """
        parameters:
        - data: data returned by process_data()
        returns:
        - (template, context) tuple
          - template is a Jinja template to use for rendering config
          options
          - context is a dictionary of context variables to use for
          rendering the form
        """
        return self.view_template, {'gradients': data['gradients'], 'image': data['image']}

    @override
    def process_data(self, input_id, input_data, output_data):
        """
        Process one inference and return data to visualize
        """
        # assume only one output and grayscale input

        output_vector = output_data[output_data.keys()[0]]
        grad = np.array([
            output_vector[0] * self.width,
            output_vector[1] * self.height])
        grad_rotated_90 = np.array([-grad[1], grad[0]])
        center = np.array([self.width / 2, self.height / 2])
        arrow = grad * (self.arrow_size / 100.)
        arrow_tip = center + arrow / 2
        arrow_tail = center - arrow / 2
        # arrow tail (anticlockwise)
        at_acw = arrow_tail + 0.1 * grad_rotated_90
        # arrow tail (clockwise)
        at_cw = arrow_tail - 0.1 * grad_rotated_90

        # draw an oriented caret
        image = PIL.Image.fromarray(input_data).convert('RGB')
        draw = PIL.ImageDraw.Draw(image)
        draw.line(
            (at_acw[0], at_acw[1], arrow_tip[0], arrow_tip[1]),
            fill=self.color)
        draw.line(
            (at_cw[0], at_cw[1], arrow_tip[0], arrow_tip[1]),
            fill=self.color)
        draw.line(
            (at_acw[0], at_acw[1], at_cw[0], at_cw[1]),
            fill=self.color)

        image_html = digits.utils.image.embed_image_html(image)
        return {'image': image_html,
                'gradients': [output_vector[0], output_vector[1]]}
