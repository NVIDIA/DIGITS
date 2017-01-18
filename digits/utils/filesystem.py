# Copyright (c) 2014-2017, NVIDIA CORPORATION.  All rights reserved.
from __future__ import absolute_import

import hashlib
import os.path
import platform
import re
import shutil


def get_tree_size(start_path):
    """
    return size (in bytes) of filesystem tree
    """
    if not os.path.exists(start_path):
        raise ValueError("Incorrect path: %s" % start_path)
    total_size = 0
    for dirpath, dirnames, filenames in os.walk(start_path):
        for f in filenames:
            fp = os.path.join(dirpath, f)
            total_size += os.path.getsize(fp)
    return total_size


def get_python_file_dst(dirname, basename):
    basename = os.path.basename(basename)
    (root, ext) = os.path.splitext(basename)
    if ext != '.py' and ext != '.pyc':
        ValueError('Python file, %s, needs .py or .pyc extension.' % basename)
    filename = os.path.join(dirname, 'digits_python_layers' + ext)
    if os.path.isfile(filename):
        ValueError('Python file, %s, already exists.' % filename)
    return filename


def copy_python_layer_file(from_client, job_dir, client_file, server_file):
    if from_client and client_file:
        filename = get_python_file_dst(job_dir, client_file.filename)
        client_file.save(filename)
    elif server_file and len(server_file) > 0:
        filename = get_python_file_dst(job_dir, server_file)
        shutil.copy(server_file, filename)


def tail(file, n=40):
    """
    Returns last n lines of text file (or all lines if the file has fewer lines)

    Arguments:
    file -- full path of that file, calling side must ensure its existence
    n -- the number of tailing lines to return
    """
    if platform.system() in ['Linux', 'Darwin']:
        import subprocess
        output = subprocess.check_output(['tail', '-n{}'.format(n), file])
    else:
        from collections import deque
        tailing_lines = deque()
        with open(file) as f:
            for line in f:
                tailing_lines.append(line)
                if len(tailing_lines) > n:
                    tailing_lines.popleft()
        output = ''.join(tailing_lines)
    return output


def dir_hash(dir_name):
    """
    Return a hash for the files in a directory tree, excluding hidden
    files and directoies. If any files are renamed, added, removed, or
    modified the hash will change.
    """
    if not os.path.isdir(dir_name):
        raise TypeError('{} is not a directory.'.format(dir_name))

    md5 = hashlib.md5()
    for root, dirs, files in os.walk(dir_name, topdown=True):
        # Skip if the root has a hidden directory in its path
        if not re.search(r'/\.', root):
            for f in files:
                # Skip if the file is hidden
                if not f.startswith('.') and not re.search(r'/\.', f):
                    # Change the hash if the file name changes
                    file_name = os.path.join(root, f)
                    md5.update(hashlib.md5(file_name).hexdigest())
                    # Change the hash if the file content changes
                    data = open(file_name, 'rb').read()
                    md5.update(hashlib.md5(data).hexdigest())
    return md5.hexdigest()
