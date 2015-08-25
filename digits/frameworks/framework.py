# Copyright (c) 2014-2015, NVIDIA CORPORATION.  All rights reserved.

class Framework(object):

    """
    Defines required methods to interact with a framework
    """

    def get_name(self):
        return self.NAME

    def get_id(self):
        return self.id

    def can_shuffle_data(self):
        return self.CAN_SHUFFLE_DATA

    def validate_network(self, data):
        return True





