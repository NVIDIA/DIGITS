# Copyright (c) 2016, NVIDIA CORPORATION.  All rights reserved.
from __future__ import absolute_import


class DataIngestionInterface(object):
    """
    A data ingestion extension
    """

    def __init__(self, **kwargs):
        # save all data there - no other fields will be persisted
        self.userdata = kwargs

        # populate instance from userdata dictionary
        for k, v in self.userdata.items():
            setattr(self, k, v)

    def encode_entry(self, entry):
        """
        Encode the entry associated with specified ID (returned by
        itemize_entries())
        Returns a tuble (feature, label)
        Data are expected in HWC format
        Color images are expected in RGB order
        """
        raise NotImplementedError

    @staticmethod
    def get_category():
        raise NotImplementedError

    @staticmethod
    def get_dataset_form():
        """
        Return a Form object with all fields required to create the dataset
        """
        raise NotImplementedError

    @staticmethod
    def get_dataset_template(form):
        """
        Parameters:
        - form: form returned by get_dataset_form(). This may be populated
           with values if the job was cloned
        return:
        - (template, context) tuple
          - template is a Jinja template to use for rendering dataset creation
          options
          - context is a dictionary of context variables to use for rendering
          the form
        """
        raise NotImplementedError

    @staticmethod
    def get_id():
        """
        Return unique ID
        """
        raise NotImplementedError

    @staticmethod
    def get_inference_form():
        """
        For later use
        """
        raise NotImplementedError

    @staticmethod
    def get_inference_template():
        """
        For later use
        """
        raise NotImplementedError

    @staticmethod
    def get_title():
        """
        Return get_title
        """
        raise NotImplementedError

    def get_user_data(self):
        """
        Return serializable user data
        The data will be persisted as part of the dataset job data
        """
        return self.userdata

    def itemize_entries(self, stage):
        """
        Return a list of entry IDs to encode
        This function is called on the main thread
        The returned list will be spread across all reader threads
        Reader threads will call encode_entry() with IDs returned by
        this function in no particular order
        """
        raise NotImplementedError
