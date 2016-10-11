# Copyright (c) 2016, NVIDIA CORPORATION.  All rights reserved.
from __future__ import absolute_import

import os

import matplotlib as mpl
import numpy as np
import PIL.Image

import digits
from digits.utils import subclass, override
from digits.utils.constants import COLOR_PALETTE_ATTRIBUTE
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
    A visualization extension to display the network output as an image
    """

    def __init__(self, dataset, **kwargs):
        # super(Visualization, self).__init__(kwargs)

        # memorize view template for later use
        extension_dir = os.path.dirname(os.path.abspath(__file__))
        self.view_template = open(
            os.path.join(extension_dir, VIEW_TEMPLATE), "r").read()

        # view options
        if kwargs['colormap'] == 'dataset':
            if not COLOR_PALETTE_ATTRIBUTE in dataset.extension_userdata:
                raise ValueError("Palette not found in dataset")
            palette = dataset.extension_userdata[COLOR_PALETTE_ATTRIBUTE]
            # assume 8-bit RGB palette and convert to N*3 numpy array
            palette = np.array(palette).reshape((len(palette)/3, 3)) / 255.
            # normalize input pixels to [0,1]
            norm = mpl.colors.Normalize(vmin=0, vmax=255)
            # create map
            cmap = mpl.colors.ListedColormap(palette)
            self.map = mpl.pyplot.cm.ScalarMappable(norm=norm, cmap=cmap)
        elif kwargs['colormap'] == 'paired':
            cmap = mpl.pyplot.cm.get_cmap('Paired')
            self.map = mpl.pyplot.cm.ScalarMappable(norm=None, cmap=cmap)
        elif kwargs['colormap'] == 'none':
            self.map = None
        else:
            raise ValueError("Unknown color map option: %s" % kwargs['colormap'])

        # memorize class labels
        if 'class_labels' in dataset.extension_userdata:
            self.class_labels = dataset.extension_userdata['class_labels']
        else:
            self.class_labels = None

        # create array to memorize all classes we found in labels
        self.found_classes = np.array([])

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

    @override
    def get_header_template(self):
        """
        Implements get_header_template() method from view extension interface
        """
        extension_dir = os.path.dirname(os.path.abspath(__file__))
        template = open(
            os.path.join(extension_dir, HEADER_TEMPLATE), "r").read()

        # show legend
        legend = []
        for cls in self.found_classes:
            # create small square image and fill it with the color
            # associated with the category ID
            if self.map:
                rgb_color = self.map.to_rgba([cls])[0, :3]*255
                image = np.zeros(shape=(14, 14, 3))
                image[:, :] = rgb_color
            else:
                image = np.full(shape=(14, 14), fill_value=cls)
            image = image.astype('uint8')
            image = digits.utils.image.embed_image_html(image)
            text = "Class #%d" % cls
            if self.class_labels:
                text = "%s (%s)" % (text, self.class_labels[int(cls)])
            legend.append({'image': image, 'text': text})
        return template, {'legend': legend}

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
        return 'image-segmentation-gl'

    @staticmethod
    def get_title():
        return 'Image Segmentation GL'

    @staticmethod
    def get_dirname():
        return 'imageSegmentationGL'

    @override
    def get_view_template(self, data):
        """
        returns:
        - (template, context) tuple
          - template is a Jinja template to use for rendering config options
          - context is a dictionary of context variables to use for rendering
          the form
        """
        width, height = data['input_image'].size
        return self.view_template, {
            'input_id': data['input_id'],
            'input_image': digits.utils.image.embed_image_html(data['input_image']),
            'output_image': digits.utils.image.embed_image_html(data['output_image']),
            'width': width,
            'height': height,
        }

    @override
    def process_data(self, input_id, input_data, output_data, ground_truth=None):
        """
        Process one inference and return data to visualize
        """
        # assume the only output is a CHW image where C is the number
        # of classes, H and W are the height and width of the image
        data = output_data[output_data.keys()[0]].astype('float32')
        # retain only the top class for each pixel
        data = np.argmax(data, axis=0)

        # remember the classes we found
        found_classes = np.unique(data)
        self.found_classes = np.unique(np.concatenate(
            (self.found_classes, found_classes)))

        # convert using color map (assume 8-bit output)
        if self.map:
            data = self.map.to_rgba(data)*255
            # keep RGB values only, remove alpha channel
            data = data[:, :, 0:3]

        # convert to uint8
        data = data.astype('uint8')
        # convert to PIL image
        output_image = PIL.Image.fromarray(data)
        output_image.format = 'png'

        input_image = PIL.Image.fromarray(input_data)
        input_image.format = 'png'

        return {
            'input_id': input_id,
            'input_image': input_image,
            'output_image': output_image,
        }
