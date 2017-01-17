# Copyright (c) 2014-2017, NVIDIA CORPORATION.  All rights reserved.
from __future__ import absolute_import

from wtforms import validators

from ..forms import ModelForm
from digits import utils


class ImageModelForm(ModelForm):
    """
    Defines the form used to create a new ImageModelJob
    """

    crop_size = utils.forms.IntegerField(
        'Crop Size',
        validators=[
            validators.NumberRange(min=1),
            validators.Optional()
        ],
        tooltip=("If specified, during training a random square crop will be "
                 "taken from the input image before using as input for the network.")
    )

    use_mean = utils.forms.SelectField(
        'Subtract Mean',
        choices=[
            ('none', 'None'),
            ('image', 'Image'),
            ('pixel', 'Pixel'),
        ],
        default='image',
        tooltip="Subtract the mean file or mean pixel for this dataset from each image."
    )

    aug_flip = utils.forms.SelectField(
        'Flipping',
        choices=[
            ('none', 'None'),
            ('fliplr', 'Horizontal'),
            ('flipud', 'Vertical'),
            ('fliplrud', 'Horizontal and/or Vertical'),
        ],
        default='none',
        tooltip="Randomly flips each image during batch preprocessing."
    )

    aug_quad_rot = utils.forms.SelectField(
        'Quadrilateral Rotation',
        choices=[
            ('none', 'None'),
            ('rot90', '0, 90 or 270 degrees'),
            ('rot180', '0 or 180 degrees'),
            ('rotall', '0, 90, 180 or 270 degrees.'),
        ],
        default='none',
        tooltip="Randomly rotates (90 degree steps) each image during batch preprocessing."
    )

    aug_rot = utils.forms.IntegerField(
        'Rotation (+- deg)',
        default=0,
        validators=[
            validators.NumberRange(min=0, max=180)
        ],
        tooltip="The uniform-random rotation angle that will be performed during batch preprocessing."
    )

    aug_scale = utils.forms.FloatField(
        'Rescale (stddev)',
        default=0,
        validators=[
            validators.NumberRange(min=0, max=1)
        ],
        tooltip=("Retaining image size, the image is rescaled with a "
                 "+-stddev of this parameter. Suggested value is 0.07.")
    )

    aug_noise = utils.forms.FloatField(
        'Noise (stddev)',
        default=0,
        validators=[
            validators.NumberRange(min=0, max=1)
        ],
        tooltip=("Adds AWGN (Additive White Gaussian Noise) during batch "
                 "preprocessing, assuming [0 1] pixel-value range. Suggested value is 0.03.")
    )

    aug_hsv_use = utils.forms.BooleanField(
        'HSV Shifting',
        default=False,
        tooltip=("Augmentation by normal-distributed random shifts in HSV "
                 "color space, assuming [0 1] pixel-value range."),
    )
    aug_hsv_h = utils.forms.FloatField(
        'Hue',
        default=0.02,
        validators=[
            validators.NumberRange(min=0, max=0.5)
        ],
        tooltip=("Standard deviation of a shift that will be performed during "
                 "preprocessing, assuming [0 1] pixel-value range.")
    )
    aug_hsv_s = utils.forms.FloatField(
        'Saturation',
        default=0.04,
        validators=[
            validators.NumberRange(min=0, max=0.5)
        ],
        tooltip=("Standard deviation of a shift that will be performed during "
                 "preprocessing, assuming [0 1] pixel-value range.")
    )
    aug_hsv_v = utils.forms.FloatField(
        'Value',
        default=0.06,
        validators=[
            validators.NumberRange(min=0, max=0.5)
        ],
        tooltip=("Standard deviation of a shift that will be performed during "
                 "preprocessing, assuming [0 1] pixel-value range.")
    )
