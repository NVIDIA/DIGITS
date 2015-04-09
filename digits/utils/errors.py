# Copyright (c) 2014-2015, NVIDIA CORPORATION.  All rights reserved.

# This class contains all the DIGITS user-defined exceptions

class DigitsError(Exception):
    def __init__(self, value):
        self.value = value
    def __str__(self):
        return self.value

class DeleteError(DigitsError):
    def __init__(self, value):
        DigitsError.__init__(self, value)
