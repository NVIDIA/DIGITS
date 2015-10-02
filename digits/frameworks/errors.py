# Copyright (c) 2015, NVIDIA CORPORATION.  All rights reserved.

from digits.utils import subclass, override

@subclass
class Error(Exception):
    pass

@subclass
class BadNetworkError(Error):
    """
    Errors that occur when validating a network
    """
    def __init__(self, message):
        self.message = message

    def __str__(self):
        return repr(self.message)

@subclass
class NetworkVisualizationError(Error):
    """
    Errors that occur when validating a network
    """
    def __init__(self, message):
        self.message = message

    def __str__(self):
        return repr(self.message)

@subclass
class InferenceError(Error):
    """
    Errors that occur during inference
    """
    def __init__(self, message):
        self.message = message

    def __str__(self):
        return repr(self.message)

