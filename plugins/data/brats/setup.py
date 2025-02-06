# Copyright (c) 2016, NVIDIA CORPORATION.  All rights reserved.

import os
from setuptools import setup, find_packages

from digits.extensions.data import GROUP as DIGITS_PLUGIN_GROUP


# Utility function to read the README file.
def read(fname):
    return open(os.path.join(os.path.dirname(__file__), fname)).read()


setup(
    name="digits_brats_data_plugin",
    version="0.0.1",
    author="Greg Heinrich",
    description=("A data ingestion plugin for the BRATS dataset"),
    long_description=read('README'),
    license="BSD",
    packages=find_packages(),
    entry_points={
        DIGITS_PLUGIN_GROUP: [
            'class=digitsDataPluginBrats:DataIngestion',
        ]},
    include_package_data=True,
    install_requires=['SimpleITK'],
)
