# Copyright (c) 2016-2017, NVIDIA CORPORATION.  All rights reserved.
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
    def get_default_visibility():
        """
        Return whether to show extension in GUI (can be overridden through
        DIGITS configuration options)
        """
        return True

    def get_header_template(self):
        """
        This returns the content to be rendered at the top of the result
        page. This may include a summary of the job as well as utility
        functions and scripts to use from "view" templates.
        By default this method returns (None, None), an indication that there
        is no header to display. This method may be overridden in sub-classes
        to show more elaborate content.
        Returns:
        - (template, context) tuple
          - template is a Jinja template to use for rendering the header,
          or None if there is no header to display
          - context is a dictionary of context variables to use for rendering
          the form
        """
        return None, None

    def get_ng_templates(self):
        """
        This returns the angularjs content defining the app and maybe a
        large scope controller. By default this method returns None,
        an indication that there is no angular header. This method
        may be overridden in sub-classes to implement the app and controller.

        This header html protion will likely be of the form:
        <script>
        ...
        </script>
        <div ng-app="my_app">
            <div ng-controller="my_controller">

        and the footer needs to close any open divs:
            </div>
        </div>
        Returns:
        - (header, footer) tuple
          - header is the html text defining the angular app and adding the ng-app div,
          or None if there is no angular app defined
          - footer is the html text closing any open divs from the header
          or None if there is no angular app defined
        """
        return None, None

    @staticmethod
    def get_id():
        """
        Returns a unique ID
        """
        raise NotImplementedError

    @staticmethod
    def get_title():
        """
        Returns a title
        """
        raise NotImplementedError

    @staticmethod
    def get_dirname():
        """
        Returns the extension's dirname for building a path to the
        extension's static files
        """
        raise NotImplementedError

    def get_view_template(self, data):
        """
        The view template shows the visualization of one inference output.
        In the case of multiple inference, this method is called once per
        input sample.
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
            input_id,
            input_data,
            inference_data,
            ground_truth=None):
        """
        Process one inference output
        Parameters:
        - input_id: index of input sample
        - input_data: input to the network
        - inference_data: network output
        - ground_truth: Ground truth. Format is application specific.
          None if absent.
        Returns:
        - an object reprensenting the processed data
        """
        raise NotImplementedError
