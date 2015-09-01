# Copyright (c) 2014-2015, NVIDIA CORPORATION.  All rights reserved.

class DigitsError(Exception):
    """
    DIGITS custom exception
    """
    pass

class DeleteError(DigitsError):
    """
    Errors that occur when deleting a job
    """
    pass

class LoadImageError(DigitsError):
    """
    Errors that occur while loading an image
    """
    pass

