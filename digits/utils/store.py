# Copyright (c) 2016-2017, NVIDIA CORPORATION.  All rights reserved.
from __future__ import absolute_import

from HTMLParser import HTMLParser
import time


class StoreCache():

    def __init__(self, ttl=86400):
        self.expiration_time = time.time() + ttl
        self.ttl = ttl
        self.cache = None

    def reset(self):
        self.expiration_time = time.time() + self.ttl
        self.cache = None

    def read(self):
        if self.expiration_time < time.time():
            self.reset()
        return self.cache

    def write(self, data):
        self.expiration_time = time.time() + self.ttl
        self.cache = data


class StoreParser(HTMLParser):

    def __init__(self):
        HTMLParser.__init__(self)
        self.starting = False
        self.dirs = list()
        self.reset()

    def read(self, data):
        self.reset()
        self.clean()
        self.feed(data)

    def clean(self):
        pass

    def handle_starttag(self, tag, attrs):
        if tag == 'td' or tag == 'a':
            self.starting = True

    def handle_endtag(self, tag):
        if tag == 'td' or tag == 'a':
            self.starting = False

    def handle_data(self, data):
        if self.starting and data[-1] == '/':
            self.dirs.append(data)

    def get_child_dirs(self):
        return self.dirs
