# Copyright (c) 2016-2017, NVIDIA CORPORATION.  All rights reserved.
from __future__ import absolute_import

import os
from urlparse import urlparse

from . import option_list


def validate(value):
    if value == '':
        return value
    valid_url_list = list()
    if isinstance(value, str):
        url_list = value.split(',')
        for url in url_list:
            if url is not None and urlparse(url).scheme != "" and not os.path.exists(url):
                valid_url_list.append(url)
            else:
                raise ValueError('"%s" is not a valid URL' % url)
    return ','.join(valid_url_list)


def load_url_list():
    """
    Return Model Store URL's as a list
    Verify if each URL is valid
    """
    if 'DIGITS_MODEL_STORE_URL' in os.environ:
        url_list = os.environ['DIGITS_MODEL_STORE_URL']
    else:
        url_list = ""

    return validate(url_list).split(',')

option_list['model_store'] = {
    'url_list': load_url_list()
}
