# Copyright (c) 2016-2017, NVIDIA CORPORATION.  All rights reserved.

import os
from setuptools import setup, find_packages

from digits.extensions.data import GROUP as DIGITS_PLUGIN_GROUP


# Utility function to read the README file.
def read(fname):
    return open(os.path.join(os.path.dirname(__file__), fname)).read()


setup(
    name="digits_image_gradients_data_plugin",
    version="0.0.1",
    author="Greg Heinrich",
    description=("A data ingestion plugin for image gradients"),
    long_description=read('README'),
    license="Apache",
    packages=find_packages(),
    entry_points={
        DIGITS_PLUGIN_GROUP: [
            'class=digitsDataPluginImageGradients:DataIngestion',
        ]
    },
    include_package_data=True,
)
