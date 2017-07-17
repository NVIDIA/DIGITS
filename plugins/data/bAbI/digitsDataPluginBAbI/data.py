# Copyright (c) 2016, NVIDIA CORPORATION.  All rights reserved.
from __future__ import absolute_import

import os

from digits.utils import subclass, override, constants
from digits.extensions.data.interface import DataIngestionInterface
from .forms import DatasetForm, InferenceForm
from . import utils


DATASET_TEMPLATE = "templates/dataset_template.html"
INFERENCE_TEMPLATE = "templates/inference_template.html"


@subclass
class DataIngestion(DataIngestionInterface):
    """
    A data ingestion extension for the bAbI dataset
    """

    def __init__(self, is_inference_db=False, **kwargs):
        super(DataIngestion, self).__init__(**kwargs)

        self.userdata['is_inference_db'] = is_inference_db

        if 'train_text_data' not in self.userdata:
            # get task ID
            try:
                task_id = int(self.task_id)
            except:
                task_id = None
            self.userdata['task_id'] = task_id

            # get data - this doesn't scale well to huge datasets but this makes it
            # straightforard to create a mapping of words to indices and figure out max
            # dimensions of stories and sentences
            self.userdata['train_text_data'] = utils.parse_folder_phase(
                self.story_folder, task_id, train=True)
            self.userdata['stats'] = utils.get_stats(self.userdata['train_text_data'])

    @override
    def encode_entry(self, entry):
        stats = self.userdata['stats']
        return utils.encode_sample(entry, stats['word_map'], stats['sentence_size'], stats['story_size'])

    @staticmethod
    @override
    def get_category():
        return "Text"

    @staticmethod
    @override
    def get_id():
        return "text-babi"

    @staticmethod
    @override
    def get_dataset_form():
        return DatasetForm()

    @staticmethod
    @override
    def get_dataset_template(form):
        """
        parameters:
        - form: form returned by get_dataset_form(). This may be populated
           with values if the job was cloned
        return:
        - (template, context) tuple
          - template is a Jinja template to use for rendering dataset creation
          options
          - context is a dictionary of context variables to use for rendering
          the form
        """
        extension_dir = os.path.dirname(os.path.abspath(__file__))
        template = open(os.path.join(extension_dir, DATASET_TEMPLATE), "r").read()
        context = {'form': form}
        return (template, context)

    @override
    def get_inference_form(self):
        return InferenceForm()

    @staticmethod
    @override
    def get_inference_template(form):
        extension_dir = os.path.dirname(os.path.abspath(__file__))
        template = open(os.path.join(extension_dir, INFERENCE_TEMPLATE), "r").read()
        context = {'form': form}
        return (template, context)

    @staticmethod
    @override
    def get_title():
        return "bAbI"

    @override
    def itemize_entries(self, stage):
        entries = []
        if not self.userdata['is_inference_db']:
            data = self.userdata['train_text_data']
            n_val_entries = int(len(data)*self.pct_val/100)
            if stage == constants.TRAIN_DB:
                entries = data[n_val_entries:]
            elif stage == constants.VAL_DB:
                entries = data[:n_val_entries]
        elif stage == constants.TEST_DB:
            if not bool(self.snippet):
                raise ValueError("You must write a story and a question")
            entries = utils.parse_lines(str(self.snippet).splitlines())

        return entries
