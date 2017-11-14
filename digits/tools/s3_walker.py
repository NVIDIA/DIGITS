#!/usr/bin/env python
# -*- coding: utf-8 -*-

from boto.s3.connection import S3Connection
from boto.s3.key import Key
from boto.s3.prefix import Prefix
import logging
from digits import utils, log  # noqa

logger = logging.getLogger('digits.tools.s3_walker')


class S3Walker(object):

    def __init__(self, endpoint, accesskey, secretkey):
        self.is_secure = endpoint.find('https://') > -1
        self.host = endpoint[endpoint.index('://')+3:]
        has_port = self.host.find(':') > -1
        if has_port:
            self.port = int(self.host[self.host.index(':')+1:])
            self.host = self.host[:self.host.index(':')]
        else:
            if self.is_secure:
                self.port = 443
            else:
                self.port = 80
        self.accesskey = accesskey
        self.secretkey = secretkey
        self.conn = None

        logger.info('Host %s is secure: %s port: %s.' % (self.host, str(self.is_secure), str(self.port)))

    def connect(self):

        self.conn = S3Connection(aws_access_key_id=self.accesskey, aws_secret_access_key=self.secretkey,
                                 is_secure=self.is_secure, host=self.host, port=self.port)

    def head(self, bucket, key):

        b = self.conn.get_bucket(bucket)
        return b.get_key(key)

    def get(self, bucket, key, filename):

        k = self.head(bucket, key)
        k.get_contents_to_filename(filename)

    def get_as_string(self, bucket, key):
        k = self.head(bucket, key)
        return k.get_contents_as_string()

    def get_meta(self, bucket, key, meta):
        value = None

        k = self.head(bucket, key)
        if k is not None:
            value = k.get_metadata(meta)

        return value

    def put(self, bucket, key, filename):

        b = self.conn.get_bucket(bucket)
        k = Key(b)
        k.key = key
        k.set_contents_from_filename(filename)

    def listbucket(self, bucket, prefix='', max_size=1000, marker='', with_prefix=False):

        logger.info('listing bucket with prefix = ' + prefix + ', with_prefix = ' + str(with_prefix))

        b = self.conn.get_bucket(bucket)
        resultset = b.list(prefix=prefix, delimiter='/', marker=marker)
        keys = []
        for key in resultset:
            # returns only Keys
            if isinstance(key, Key):
                keys.append(key.key)
            elif isinstance(key, Prefix) and with_prefix:
                keys.append(key.name)
            if len(keys) >= max_size:
                break
        if len(keys) > 0:
            logger.info('retrieved ' + str(len(keys)) + ' keys from ' + keys[0] + ' to ' + keys[-1])

        return keys

    def create_bucket(self, bucket):
        return self.conn.create_bucket(bucket)
