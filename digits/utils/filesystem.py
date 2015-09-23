# Copyright (c) 2014-2015, NVIDIA CORPORATION.  All rights reserved.

import os.path, shutil
import werkzeug.exceptions

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
    filename = os.path.join(dirname, basename)
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
