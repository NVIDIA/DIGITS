# Copyright (c) 2016-2017, NVIDIA CORPORATION.  All rights reserved.
from __future__ import absolute_import


class DataIngestionInterface(object):
    """
    A data ingestion extension
    """

    def __init__(self, is_inference_db=False, **kwargs):
        """
        Initialize the data ingestion extension
        Parameters:
        - is_inference_db: boolean value, indicates whether the database is
          created for inference. If this is true then the extension needs to
          use the data from the inference form and create a database only for
          the test phase (stage == constants.TEST_DB)
        - kwargs: dataset form fields
        """
        # save all data there - no other fields will be persisted
        self.userdata = kwargs

        # populate instance from userdata dictionary
        for k, v in self.userdata.items():
            setattr(self, k, v)

    def encode_entry(self, entry):
        """
        Encode the entry associated with specified ID (returned by
        itemize_entries())
        Returns a list of (feature, label) tuples, or a single tuple
        if there is only one sample for the entry.
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

    def get_inference_form(self):
        """
        Return a Form object with all fields required to create an inference dataset
        """
        return None

    @staticmethod
    def get_inference_template(form):
        """
        Parameters:
        - form: form returned by get_inference_form().
        return:
        - (template, context) tuple
          - template is a Jinja template to use for rendering the inference form
          - context is a dictionary of context variables to use for rendering
          the form
        """
        return (None, None)

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
