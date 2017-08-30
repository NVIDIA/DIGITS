# Copyright (c) 2014-2017, NVIDIA CORPORATION.  All rights reserved.
from __future__ import absolute_import

import time


class Status():
    """
    A little class to store the state of Jobs and Tasks
    It's pickle-able!
    """

    # Enum-like attributes

    INIT = 'I'
    WAIT = 'W'
    RUN = 'R'
    DONE = 'D'
    ABORT = 'A'
    ERROR = 'E'

    def __init__(self, val):
        self.set_dict(val)

    def __str__(self):
        return self.val

    # Pickling

    def __getstate__(self):
        return self.val

    def __setstate__(self, state):
        self.set_dict(state)

    # Operators

    def __eq__(self, other):
        if type(other) == type(self):
            return self.val == other.val
        elif type(other) == str:
            return self.val == other
        else:
            return False

    def __ne__(self, other):
        if type(other) == type(self):
            return self.val != other.val
        elif type(other) == str:
            return self.val != other
        else:
            return True

    # Member functions

    def set_dict(self, val):
        self.val = val
        if val == self.INIT:
            self.name = 'Initialized'
            self.css = 'warning'
        elif val == self.WAIT:
            self.name = 'Waiting'
            self.css = 'warning'
        elif val == self.RUN:
            self.name = 'Running'
            self.css = 'info'
        elif val == self.DONE:
            self.name = 'Done'
            self.css = 'success'
        elif val == self.ABORT:
            self.name = 'Aborted'
            self.css = 'warning'
        elif val == self.ERROR:
            self.name = 'Error'
            self.css = 'danger'
        else:
            self.name = '?'
            self.css = 'default'

    def is_running(self):
        return self.val in (self.INIT, self.WAIT, self.RUN)


class StatusCls(object):
    """
    A class that stores a history of Status updates
    Child classes can declare the on_status_update() callback
    """

    def __init__(self):
        self.progress = 0
        self.status_history = []
        self.status = Status.INIT

    @property
    def status(self):
        if len(self.status_history) > 0:
            return self.status_history[-1][0]
        else:
            return Status.INIT

    @status.setter
    def status(self, value):
        if isinstance(value, str):
            value = Status(value)
        assert isinstance(value, Status)

        if self.status_history and value == self.status_history[-1][0]:
            return

        self.status_history.append((value, time.time()))

        # Remove WAIT status if waited for less than 1 second
        if value == Status.RUN and len(self.status_history) >= 2:
            curr = self.status_history[-1]
            prev = self.status_history[-2]
            if prev[0] == Status.WAIT and (curr[1] - prev[1]) < 1:
                self.status_history.pop(-2)

        # If the status is Done, then force the progress to 100%
        if value == Status.DONE:
            self.progress = 1.0
            if hasattr(self, 'emit_progress_update'):
                self.emit_progress_update()

        # Don't invoke callback for INIT
        if value != Status.INIT:
            if hasattr(self, 'on_status_update'):
                self.on_status_update()
