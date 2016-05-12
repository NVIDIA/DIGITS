# Copyright (c) 2016, NVIDIA CORPORATION.  All rights reserved.
from __future__ import absolute_import


class VisualizationInterface(object):
    """
    A visualization extension
    """

    def __init__(self, **kwargs):
        pass

    @staticmethod
    def get_config_form():
        """
        Return a form to be used to configure visualization options
        """
        raise NotImplementedError

    @staticmethod
    def get_config_template(form):
        """
        The config template shows a form with view config options
        Parameters:
        - form: form returned by get_config_form(). This may be populated
          with values if the job was cloned
        Returns:
        - (template, context) tuple
          - template is a Jinja template to use for rendering config options
          - context is a dictionary of context variables to use for rendering
          the form
        """
        raise NotImplementedError

    @staticmethod
    def get_id():
        """
        Returns a unique ID
        """
        raise NotImplementedError

    def get_summary_template(self):
        """
        This returns a summary of the job. This method is called after all
        entries have been processed.
        Returns:
        - (template, context) tuple
          - template is a Jinja template to use for rendering the summary,
          or None if there is no summary to display
          - context is a dictionary of context variables to use for rendering
          the form
        """
        return None, None

    @staticmethod
    def get_title():
        """
        Returns a title
        """
        raise NotImplementedError

    def get_view_template(self, data):
        """
        The view template shows the visualization of one inference output
        Parameters:
        - data: the data returned by process_data()
        Returns:
        - (template, context) tuple
          - template is a Jinja template to use for rendering config options
          - context is a dictionary of context variables to use for rendering
          the form
        """
        raise NotImplementedError

    def process_data(
            self,
            dataset,
            input_data,
            inference_data,
            ground_truth=None):
        """
        Process one inference output
        Parameters:
        - dataset: dataset used during training
        - input_data: input to the network
        - inference_data: network output
        - ground_truth: Ground truth. Format is application specific.
          None if absent.
        Returns:
        - an object reprensenting the processed data
        """
        raise NotImplementedError
