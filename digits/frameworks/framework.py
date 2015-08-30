# Copyright (c) 2015, NVIDIA CORPORATION.  All rights reserved.

class Framework(object):

    """
    Defines required methods to interact with a framework
    """

    def get_name(self):
        """
        return self-descriptive name
        """
        return self.NAME

    def get_id(self):
        """
        return unique id of framework instance
        """
        return self.framework_id

    def can_shuffle_data(self):
        """
        return whether framework can shuffle input data during training
        """
        return self.CAN_SHUFFLE_DATA

    def validate_network(self, data):
        """
        validate a network (must be implemented in child class)
        """
        raise NotImplementedError('Please implement me')

    def create_train_task(self, **kwargs):
        """
        create train task
        """
        raise NotImplementedError('Please implement me')

    def get_standard_network_desc(self, network):
        """
        return text description of network
        """
        raise NotImplementedError('Please implement me')

    def get_network_from_desc(self, network_desc):
        """
        return network object from a string representation
        """
        raise NotImplementedError('Please implement me')

    def get_network_from_previous(self, previous_network):
        """
        return new instance of network from previous network
        """
        raise NotImplementedError('Please implement me')

    def get_network_visualization(self, desc):
        """
        return visualization of network
        """
        raise NotImplementedError('Please implement me')





