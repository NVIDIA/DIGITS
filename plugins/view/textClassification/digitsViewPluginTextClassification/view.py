# Copyright (c) 2016-2017, NVIDIA CORPORATION.  All rights reserved.
from __future__ import absolute_import

import os

import numpy as np

from digits.utils import subclass, override
from digits.extensions.view.interface import VisualizationInterface
from .forms import ConfigForm


CONFIG_TEMPLATE = "templates/config_template.html"
VIEW_TEMPLATE = "templates/view_template.html"


@subclass
class Visualization(VisualizationInterface):
    """
    A visualization extension to display the output
    of a text classification network
    """

    def __init__(self, dataset, **kwargs):
        """
        Init
        """
        # memorize view template for later use
        extension_dir = os.path.dirname(os.path.abspath(__file__))
        self.view_template = open(
            os.path.join(extension_dir, VIEW_TEMPLATE), "r").read()

        # memorize class labels
        if 'class_labels' in dataset.extension_userdata:
            self.class_labels = dataset.extension_userdata['class_labels']
        else:
            self.class_labels = None

        # memorize alphabet
        if 'alphabet' in dataset.extension_userdata:
            self.alphabet = dataset.extension_userdata['alphabet']
            self.alphabet_len = len(self.alphabet)
        else:
            raise RuntimeError("No alphabet found in dataset")

        # view options
        self.max_classes = kwargs['max_classes']

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
        return "text-classification"

    @staticmethod
    def get_title():
        return "Text Classification"

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
        return self.view_template, {'input': data['input'],
                                    'predictions': data['predictions']}

    @override
    def process_data(self, input_id, input_data, output_data):
        """
        Process one inference and return data to visualize
        """
        # convert input data to a string of characters
        input_characters = []
        for idx in input_data[0]:
            c = self.alphabet[idx - 1] if idx < self.alphabet_len else '.'
            input_characters.append(c)
        input_string = ''.join(input_characters)

        # assume the only output is from a probability distribution
        scores = output_data[output_data.keys()[0]].astype('float32')

        if np.max(scores) < 0:
            # terminal layer is a logsoftmax
            scores = np.exp(scores)

        indices = (-scores).argsort()
        predictions = [(self.class_labels[i] if self.class_labels else '#%d' % i,
                        round(100.0 * scores[i], 2)) for i in indices[:self.max_classes]]

        return {'input': input_string, 'predictions': predictions}
