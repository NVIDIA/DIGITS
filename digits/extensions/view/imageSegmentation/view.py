# Copyright (c) 2016, NVIDIA CORPORATION.  All rights reserved.
from __future__ import absolute_import

import json
import os

import matplotlib as mpl
import numpy as np
import PIL.Image
import skfmm

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
            palette = np.array(palette).reshape((len(palette)/3,3)) / 255.
            # normalize input pixels to [0,1]
            norm = mpl.colors.Normalize(vmin=0,vmax=255)
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

    def get_legend_for(self, found_classes, skip_classes=[]):
        """
        Return the legend color image squares and text for each class
        :param found_classes: list of class indices
        :param skip_classes: list of class indices to skip
        :return: list of dicts of text hex_color for each class
        """
        legend = []
        for c in (x for x in found_classes if x not in skip_classes):
            # create hex color associated with the category ID
            if self.map:
                rgb_color = self.map.to_rgba([c])[0, :3]
                hex_color = mpl.colors.rgb2hex(rgb_color)
            else:
                # make a grey scale hex color
                h = hex(int(c)).split('x')[1].zfill(2)
                hex_color = '#%s%s%s' % (h, h, h)

            if self.class_labels:
                text = self.class_labels[int(c)]
            else:
                text = "Class #%d" % c

            legend.append({'index':c, 'text': text, 'hex_color': hex_color})
        return legend

    @override
    def get_header_template(self):
        """
        Implements get_header_template() method from view extension interface
        """
        extension_dir = os.path.dirname(os.path.abspath(__file__))
        template = open(
            os.path.join(extension_dir, HEADER_TEMPLATE), "r").read()

        return template, {}

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
        return 'image-segmentation'

    @staticmethod
    def get_title():
        return 'Image Segmentation'

    @staticmethod
    def get_dirname():
        return 'imageSegmentation'

    @override
    def get_view_template(self, data):
        """
        returns:
        - (template, context) tuple
          - template is a Jinja template to use for rendering config options
          - context is a dictionary of context variables to use for rendering
          the form
        """
        return self.view_template, {
            'input_id': data['input_id'],
            'input_image': digits.utils.image.embed_image_html(data['input_image']),
            'fill_image': digits.utils.image.embed_image_html(data['fill_image']),
            'mask_image': digits.utils.image.embed_image_html(data['mask_image']),
            'legend': data['legend'],
            'class_data': json.dumps(data['class_data'].tolist()),
        }

    @override
    def process_data(self, input_id, input_data, output_data):
        """
        Process one inference and return data to visualize
        """
        # assume the only output is a CHW image where C is the number
        # of classes, H and W are the height and width of the image
        class_data = output_data[output_data.keys()[0]].astype('float32')
        # retain only the top class for each pixel
        class_data = np.argmax(class_data, axis=0).astype('uint8')

        # remember the classes we found
        found_classes = np.unique(class_data)

        # convert using color map (assume 8-bit output)
        if self.map:
            fill_data = (self.map.to_rgba(class_data)*255).astype('uint8')
        else:
            fill_data = np.ndarray((class_data.shape[0], class_data.shape[1], 4), dtype='uint8')
            for x in xrange(3):
                fill_data[:, :, x] = class_data.copy()

        # Assuming that class 0 is the background
        mask = np.greater(class_data, 0)
        fill_data[:, :, 3] = mask * 255

        # Black mask of non-segmented pixels
        mask_data = np.zeros(fill_data.shape, dtype='uint8')
        mask_data[:, :, 3] = (1 - mask) * 255

        # Generate outlines around segmented classes
        if len(found_classes) > 1:
            # Assuming that class 0 is the background.
            line_mask = np.zeros(class_data.shape, dtype=bool)
            for c in (x for x in found_classes if x != 0):
                c_mask = np.equal(class_data, c)
                # Find the signed distance from the zero contour
                distance = skfmm.distance(c_mask.astype('float32') - 0.5)
                # Accumulate the mask for all classes
                line_width = 3
                line_mask |= c_mask & np.less(distance, line_width)

            # add the outlines to the input image
            for x in xrange(3):
                input_data[:, :, x] = (input_data[:, :, x] * (1 - line_mask) +
                                       fill_data[:, :, x] * line_mask)

        # Input image with outlines
        input_image = PIL.Image.fromarray(input_data)
        input_image.format = 'png'

        # Fill image
        fill_image = PIL.Image.fromarray(fill_data)
        fill_image.format = 'png'

        # Mask image
        mask_image = PIL.Image.fromarray(mask_data)
        mask_image.format = 'png'

        # legend for this instance
        legend = self.get_legend_for(found_classes, skip_classes=[0])

        return {
            'input_id': input_id,
            'input_image': input_image,
            'fill_image': fill_image,
            'mask_image': mask_image,
            'legend': legend,
            'class_data': class_data,
        }
