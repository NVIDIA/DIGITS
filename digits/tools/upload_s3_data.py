#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys
import os
import ConfigParser
from s3_walker import S3Walker

config = ConfigParser.RawConfigParser()
config.read('upload_config.cfg')
endpoint = config.get('S3 Config', 'endpoint')
accesskey = config.get('S3 Config', 'accesskey')
secretkey = config.get('S3 Config', 'secretkey')
bucket_name = config.get('S3 Config', 'bucket')
path_prefix = config.get('S3 Config', 'prefix')
if not (path_prefix.endswith('/')):
    path_prefix += '/'

# mnist
# - train
# -- 0 ... 9
# --- XXX.png
try:
    mnist_folder = sys.argv[1]
except IndexError:
    print('mnist folder should be passed')
    sys.exit(1)

walker = S3Walker(endpoint, accesskey, secretkey)
walker.connect()

# Create bucket
print('Creating bucket')
walker.create_bucket(bucket_name)

mnist_train_folder = os.path.join(mnist_folder, 'train')
digits = os.listdir(mnist_train_folder)
for digit in digits:
    digit_folder = os.path.join(mnist_train_folder, digit)
    if os.path.isfile(digit_folder):
        continue
    files = os.listdir(digit_folder)
    for f in files:
        if not f.endswith('.png'):
            continue
        file = os.path.join(digit_folder, f)
        key = path_prefix + file[file.index('train'):]
        walker.put(bucket_name, key, file)
        print('uploaded ' + file + ' ==> ' + key)
