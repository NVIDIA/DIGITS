# Copyright (c) 2014-2017, NVIDIA CORPORATION.  All rights reserved.
from __future__ import absolute_import

import datetime
from ..job import ModelJob
from digits.utils import subclass, override

# NOTE: Increment this every time the pickled object changes
PICKLE_VERSION = 1


@subclass
class ImageModelJob(ModelJob):
    """
    A Job that creates an image model
    """

    def __init__(self, **kwargs):
        """
        """
        super(ImageModelJob, self).__init__(**kwargs)
        self.pickver_job_model_image = PICKLE_VERSION

    @override
    def json_dict(self, verbose=False, epoch=-1):
        d = super(ImageModelJob, self).json_dict(verbose)
        task = self.train_task()
        creation_time = str(datetime.datetime.fromtimestamp(self.status_history[0][1]))

        d.update({
            "job id": self.id(),
            "creation time": creation_time,
            "username": self.username,
        })

        d.update(task.get_task_stats(epoch))
        return d
