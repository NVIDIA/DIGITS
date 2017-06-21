# Copyright (c) 2016, NVIDIA CORPORATION.  All rights reserved.
from __future__ import absolute_import

import os

# Find the best implementation available
try:
    from cStringIO import StringIO
except ImportError:
    from StringIO import StringIO
import pickle

import imageio
import numpy as np
import PIL.Image
import PIL.ImageDraw

import digits
from digits.utils import subclass, override
from digits.extensions.view.interface import VisualizationInterface
from .forms import ConfigForm


CONFIG_TEMPLATE = "templates/config_template.html"
HEADER_TEMPLATE = "templates/header_template.html"
VIEW_TEMPLATE = "templates/view_template.html"

CELEBA_ATTRIBUTES = """
                     5_o_Clock_Shadow Arched_Eyebrows Attractive Bags_Under_Eyes Bald Bangs
                     Big_Lips Big_Nose Black_Hair Blond_Hair Blurry Brown_Hair Bushy_Eyebrows
                     Chubby Double_Chin Eyeglasses Goatee Gray_Hair Heavy_Makeup High_Cheekbones
                     Male Mouth_Slightly_Open Mustache Narrow_Eyes No_Beard Oval_Face Pale_Skin
                     Pointy_Nose Receding_Hairline Rosy_Cheeks Sideburns Smiling Straight_Hair
                     Wavy_Hair Wearing_Earrings Wearing_Hat Wearing_Lipstick Wearing_Necklace
                     Wearing_Necktie Young
                    """.split()


@subclass
class Visualization(VisualizationInterface):
    """
    A visualization extension to display the output of a GAN
    """

    def __init__(self, dataset, **kwargs):
        """
        Init
        """
        # memorize view template for later use
        extension_dir = os.path.dirname(os.path.abspath(__file__))
        self.view_template = open(
            os.path.join(extension_dir, VIEW_TEMPLATE), "r").read()

        self.normalize = True
        self.grid_size = 10

        # view options
        self.task_id = kwargs['gan_view_task_id']
        self.attributes_file = kwargs['attributes_file']

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

        context = {'task_id': self.task_id,
                   'cols': range(self.grid_size),
                   'rows': range(self.grid_size),
                   'animated_image': None}

        if hasattr(self, 'animated_images'):
            # create animated gif
            string_buf = StringIO()
            fmt = "gif"
            imageio.mimsave(string_buf, self.animated_images, format=fmt)
            data = string_buf.getvalue().encode('base64').replace('\n', '')
            animated_image_html = 'data:image/%s;base64,%s' % (fmt, data)
            context['animated_image'] = animated_image_html

        return template, context

    @staticmethod
    def get_id():
        return "image-gan"

    @staticmethod
    def get_title():
        return "GAN"

    def get_image_html(self, image):
        # assume 8-bit
        if self.normalize:
            image -= image.min()
            if image.max() > 0:
                image /= image.max()
                image *= 255
        else:
            # clip
            image = image.clip(0, 255)

        # convert to uint8
        image = image.astype('uint8')

        # convert to PIL image
        channels = image.shape[2]
        if channels == 1:
            # drop channel axis
            image = PIL.Image.fromarray(image[:, :, 0])
        elif channels == 3:
            image = PIL.Image.fromarray(image)
        else:
            raise ValueError("Unhandled number of channels: %d" % channels)

        # image.save(fname)

        image_html = digits.utils.image.embed_image_html(image)

        return image_html

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
        context = {'task_id': self.task_id}
        context.update(data)
        if self.task_id in ['celeba_encoder', 'mnist_encoder']:
            context.update({'task_id': 'encoder'})
        return self.view_template, context

    @override
    def process_data(self, input_id, input_data, output_data):
        """
        Process one inference and return data to visualize
        """
        data = output_data[output_data.keys()[0]].astype('float32')

        if self.task_id == 'grid':
            col_id = int(input_id) // self.grid_size
            row_id = int(input_id) % self.grid_size
            image_html = self.get_image_html(data)

            img_size = data.shape[0]
            if img_size == 28:
                # MNIST
                if not hasattr(self, 'animated_images'):
                    self.animated_images = [None] * (self.grid_size ** 2)
                self.animated_images[row_id * self.grid_size + col_id] = data.astype('uint8')
            elif img_size == 64:
                # CelebA
                if not hasattr(self, 'animated_images'):
                    self.animated_images = [None] * (4 * self.grid_size - 4)
                print("animated: %s" % repr(self.animated_images))

                if (col_id == 0 or row_id == 0 or col_id == (self.grid_size - 1) or row_id == (self.grid_size - 1)):
                    if row_id == 0:
                        idx = col_id
                    elif col_id == (self.grid_size - 1):
                        idx = self.grid_size - 1 + row_id
                    elif row_id == (self.grid_size - 1):
                        idx = 3 * self.grid_size - 3 - col_id
                    else:
                        idx = 4 * self.grid_size - 4 - row_id
                    self.animated_images[idx] = data.astype('uint8')
                    print("set idx %d " % idx)
            else:
                raise ValueError("Unhandled image size: %d" % img_size)

            return {'image': image_html,
                    'col_id': col_id,
                    'row_id': row_id,
                    'key': input_id}
        elif self.task_id == 'mnist_encoder':
            self.z_dim = 100
            z = data[:self.z_dim]
            image = data[self.z_dim:].reshape(28, 28)
            input_data = input_data.astype('float32')
            input_data = input_data[:, :, np.newaxis]
            image = image[:, :, np.newaxis]
            image_input_html = self.get_image_html(input_data)
            image_output_html = self.get_image_html(image)
            return {'z': z,
                    'image_input': image_input_html,
                    'image_output': image_output_html,
                    'key': input_id}
        elif self.task_id == 'celeba_encoder':
            self.z_dim = 100
            z = data[:self.z_dim]
            image = data[self.z_dim:].reshape(64, 64, 3)
            input_data = input_data.astype('float32')
            image_input_html = self.get_image_html(input_data)
            image_output_html = self.get_image_html(image)
            return {'z': z,
                    'image_input': image_input_html,
                    'image_output': image_output_html,
                    'key': input_id}
        elif self.task_id == 'animation':
            image_html = self.get_image_html(data)
            if not hasattr(self, 'animated_images'):
                self.animated_images = []
            self.animated_images.append(data.astype('uint8'))
            return {'image': image_html,
                    'key': input_id}
        elif self.task_id == 'attributes':
            self.z_dim = 100
            z = data[:self.z_dim]
            input_data = input_data.astype('float32')
            image_input_html = self.get_image_html(input_data)
            image = data[self.z_dim:].reshape(64, 64, 3)
            image_output_html = self.get_image_html(image)
            with open(self.attributes_file, 'rb') as f:
                attributes_z = pickle.load(f)

            # inner_products = np.inner(z, attributes_z)
            inner_products = np.empty((40))
            for i in range(40):
                if True:
                    attr = attributes_z[i]
                    inner_products[i] = np.inner(z, attr) / np.linalg.norm(attr)
                else:
                    inner_products[i] = 0

            top_5_indices = np.argsort(inner_products)[::-1][:5]
            top_5 = [(CELEBA_ATTRIBUTES[idx], "%.2f" % inner_products[idx]) for idx in top_5_indices]
            return {'image_input': image_input_html,
                    'image_output': image_output_html,
                    'top5': top_5}
        else:
            raise ValueError("Unknown task: %s" % self.task_id)
