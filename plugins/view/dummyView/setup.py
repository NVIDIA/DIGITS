# Copyright (c) 2016, NVIDIA CORPORATION.  All rights reserved.
import os
from setuptools import setup, find_packages

from digits.extensions.view import GROUP as DIGITS_PLUGIN_GROUP

# Utility function to read the README file.
def read(fname):
    return open(os.path.join(os.path.dirname(__file__), fname)).read()

setup(
    name="digits_dummy_view_plugin",
    version="0.0.1",
    author="Greg Heinrich",
    description=("A dummy vizualisation plugin"),
    long_description=read('README'),
    license="Apache",
    packages=find_packages(),
    entry_points={
        DIGITS_PLUGIN_GROUP: [
        'visualization=dummyView:Visualization',
        ]},
    include_package_data=True,
    #data_files=[('dummyView/config_template.html','dummyView/config_template.html')],
)