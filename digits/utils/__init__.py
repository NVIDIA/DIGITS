# Copyright (c) 2014-2017, NVIDIA CORPORATION.  All rights reserved.
from __future__ import absolute_import

import inspect
from io import BlockingIOError
import locale
import math
import os
import pkg_resources
import platform
from random import uniform
from urlparse import urlparse

if not platform.system() == 'Windows':
    import fcntl
else:
    import gevent.os

HTTP_TIMEOUT = 6.05


def is_url(url):
    return url is not None and urlparse(url).scheme != "" and not os.path.exists(url)


def wait_time():
    """Wait a random number of seconds"""
    return uniform(0.3, 0.5)

# From http://code.activestate.com/recipes/578900-non-blocking-readlines/


def nonblocking_readlines(f):
    """Generator which yields lines from F (a file object, used only for
       its fileno()) without blocking.  If there is no data, you get an
       endless stream of empty strings until there is data again (caller
       is expected to sleep for a while).
       Newlines are normalized to the Unix standard.
    """
    fd = f.fileno()
    if not platform.system() == 'Windows':
        fl = fcntl.fcntl(fd, fcntl.F_GETFL)
        fcntl.fcntl(fd, fcntl.F_SETFL, fl | os.O_NONBLOCK)
    enc = locale.getpreferredencoding(False)

    buf = bytearray()
    while True:
        try:
            if not platform.system() == 'Windows':
                block = os.read(fd, 8192)
            else:
                block = gevent.os.tp_read(fd, 8192)
        except (BlockingIOError, OSError):
            yield ""
            continue

        if not block:
            if buf:
                yield buf.decode(enc)
            break

        buf.extend(block)

        while True:
            r = buf.find(b'\r')
            n = buf.find(b'\n')
            if r == -1 and n == -1:
                break

            if r == -1 or r > n:
                yield buf[:(n + 1)].decode(enc)
                buf = buf[(n + 1):]
            elif n == -1 or n > r:
                yield buf[:r].decode(enc) + '\n'
                if n == r + 1:
                    buf = buf[(r + 2):]
                else:
                    buf = buf[(r + 1):]


def subclass(cls):
    """
    Verify all @override methods
    Use a class decorator to find the method's class
    """
    for name, method in cls.__dict__.iteritems():
        if hasattr(method, 'override'):
            found = False
            for base_class in inspect.getmro(cls)[1:]:
                if name in base_class.__dict__:
                    if not method.__doc__:
                        # copy docstring
                        method.__doc__ = base_class.__dict__[name].__doc__
                    found = True
                    break
            assert found, '"%s.%s" not found in any base class' % (cls.__name__, name)
    return cls


def override(method):
    """
    Decorator implementing method overriding in python
    Must also use the @subclass class decorator
    """
    method.override = True
    return method


def sizeof_fmt(size, suffix='B'):
    """
    Return a human-readable string representation of a filesize

    Arguments:
    size -- size in bytes
    """
    try:
        size = int(size)
    except ValueError:
        return None
    if size <= 0:
        return '0 %s' % suffix

    size_name = ('', 'K', 'M', 'G', 'T', 'P', 'E', 'Z', 'Y')
    i = int(math.floor(math.log(size, 1024)))
    if i >= len(size_name):
        i = len(size_name) - 1
    p = math.pow(1024, i)
    s = size / p
    # round to 3 significant digits
    s = round(s, 2 - int(math.floor(math.log10(s))))
    if s.is_integer():
        s = int(s)
    if s > 0:
        return '%s %s%s' % (s, size_name[i], suffix)
    else:
        return '0 %s' % suffix


def parse_version(*args):
    """
    Returns a sortable version

    Arguments:
    args -- a string, tuple, or list of arguments to be joined with "."'s
    """
    v = None
    if len(args) == 1:
        a = args[0]
        if isinstance(a, tuple):
            v = '.'.join(str(x) for x in a)
        else:
            v = str(a)
    else:
        v = '.'.join(str(a) for a in args)

    if v.startswith('v'):
        v = v[1:]

    try:
        return pkg_resources.SetuptoolsVersion(v)
    except AttributeError:
        return pkg_resources.parse_version(v)


# Import the other utility functions

from . import constants, image, time_filters, errors, forms, routing, auth  # noqa
