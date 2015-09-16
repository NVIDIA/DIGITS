# Copyright (c) 2014-2015, NVIDIA CORPORATION.  All rights reserved.

import os.path

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

