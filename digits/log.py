# Copyright (c) 2014-2015, NVIDIA CORPORATION.  All rights reserved.

import sys
import os
import logging
import logging.handlers

from digits.config import config_option


DATE_FORMAT = '%Y-%m-%d %H:%M:%S'

class JobIdLogger(logging.Logger):
    def makeRecord(self, name, level, fn, lno, msg, args, exc_info, func=None, extra=None):
        """
        Customizing it to set a default value for extra['job_id']
        """
        rv = logging.LogRecord(name, level, fn, lno, msg, args, exc_info, func)
        if extra is not None:
            for key in extra:
                if (key in ["message", "asctime"]) or (key in rv.__dict__):
                    raise KeyError("Attempt to overwrite %r in LogRecord" % key)
                rv.__dict__[key] = extra[key]
        if 'job_id' not in rv.__dict__:
            rv.__dict__['job_id'] = ''
        return rv

class JobIdLoggerAdapter(logging.LoggerAdapter):
    """
    Accepts an optional keyword argument: 'job_id'

    You can use this in 2 ways:
        1. On class initialization
            adapter = JobIdLoggerAdapter(logger, {'job_id': job_id})
            adapter.debug(msg)
        2. On method invokation
            adapter = JobIdLoggerAdapter(logger, {})
            adapter.debug(msg, job_id=id)
    """
    def process(self, msg, kwargs):
        if 'job_id' in kwargs:
            if 'extra' not in kwargs:
                kwargs['extra'] = {}
            kwargs['extra']['job_id'] = ' [%s]' % kwargs['job_id']
            del kwargs['job_id']
        elif 'job_id' in self.extra:
            if 'extra' not in kwargs:
                kwargs['extra'] = {}
            kwargs['extra']['job_id'] = ' [%s]' % self.extra['job_id']
        return msg, kwargs

def setup_logging():
    socketio_logger = logging.getLogger('socketio')
    socketio_logger.addHandler(logging.StreamHandler(sys.stdout))

    # Set custom logger
    logging.setLoggerClass(JobIdLogger)

    formatter = logging.Formatter(
            fmt="%(asctime)s%(job_id)s [%(levelname)-5s] %(message)s",
            datefmt=DATE_FORMAT,
            )

    ### digits logger

    main_logger = logging.getLogger('digits')
    main_logger.setLevel(logging.DEBUG)
    # Log to stdout
    stdoutHandler = logging.StreamHandler(sys.stdout)
    stdoutHandler.setFormatter(formatter)
    stdoutHandler.setLevel(logging.DEBUG)
    main_logger.addHandler(stdoutHandler)

    ### digits.webapp logger

    if config_option('log_file'):
        webapp_logger = logging.getLogger('digits.webapp')
        webapp_logger.setLevel(logging.DEBUG)
        # Log to file
        fileHandler = logging.handlers.RotatingFileHandler(
                config_option('log_file'),
                maxBytes=(1024*1024*10), # 10 MB
                backupCount=10,
                )
        fileHandler.setFormatter(formatter)
        if config_option('log_level') == 'debug':
            fileHandler.setLevel(logging.DEBUG)
        elif config_option('log_level') == 'info':
            fileHandler.setLevel(logging.INFO)
        elif config_option('log_level') == 'warning':
            fileHandler.setLevel(logging.WARNING)
        elif config_option('log_level') == 'error':
            fileHandler.setLevel(logging.ERROR)
        elif config_option('log_level') == 'critical':
            fileHandler.setLevel(logging.CRITICAL)
        webapp_logger.addHandler(fileHandler)

        ### Useful shortcut for the webapp, which may set job_id

        return JobIdLoggerAdapter(webapp_logger, {})
    else:
        return JobIdLoggerAdapter(main_logger, {})

# Do it when this module is loaded
logger = setup_logging()

