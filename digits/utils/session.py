# Copyright (c) 2014-2015, NVIDIA CORPORATION.  All rights reserved.

from functools import update_wrapper

from flask import session, redirect, url_for

# simple session decorator
def session_required(f):
    def wrapper(*args, **kwargs):
        if 'username' not in session:
            return redirect(url_for('login'))

        return f(*args, **kwargs)

    update_wrapper(wrapper, f)
    wrapper.__name__ = f.__name__

    return wrapper
