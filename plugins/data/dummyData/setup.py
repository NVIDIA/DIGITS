import os
from setuptools import setup, find_packages

# Copyright (c) 2016, NVIDIA CORPORATION.  All rights reserved.
from digits.extensions.data import GROUP as DIGITS_PLUGIN_GROUP

# Utility function to read the README file.
def read(fname):
    return open(os.path.join(os.path.dirname(__file__), fname)).read()

setup(
    name="digits_dummy_data_plugin",
    version="0.0.1",
    author="Greg Heinrich",
    description=("A dummy data ingestion plugin"),
    long_description=read('README'),
    license="Apache",
    packages=find_packages(),
    entry_points={
        DIGITS_PLUGIN_GROUP: [
        'visualization=dummyData:DataIngestion',
        ]},
    include_package_data=True,
)