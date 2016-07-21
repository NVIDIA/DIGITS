from setuptools import setup, find_packages

import digits

setup(
    name = "digits",
    packages = find_packages(),
    version = digits.__version__
)
