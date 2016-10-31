
# Copyright (c) 2016, NVIDIA CORPORATION.  All rights reserved.
from __future__ import absolute_import

import os

import numpy as np
import SimpleITK as sitk


def encode_sample(files, filter_method='all', threshold=5000.):
    """
    return an encoded (feature, label) tuple
    """
    # get filenames
    feature_filename = files[0]
    ground_truth_filename = files[1]

    # SITK requires ASCII strings (not unicode)
    feature_filename = feature_filename.encode('ascii', 'replace')
    ground_truth_filename = ground_truth_filename.encode('ascii', 'replace')

    # load files
    feature = sitk.GetArrayFromImage(sitk.ReadImage(feature_filename))
    label = sitk.GetArrayFromImage(sitk.ReadImage(ground_truth_filename))

    # compute mean per axial slice
    means = np.mean(np.mean(label, axis=1), axis=1)

    if filter_method == 'max':
        # retain only slice with max tumor area
        max_depth = np.argmax(means)
        # extract relevant plane and reshape
        feature = feature[np.newaxis, max_depth, :]
        label = label[np.newaxis, max_depth, :]
    elif filter_method == 'threshold':
        # retain only slices with >threshold tumor pixels
        indices = np.nonzero(means > float(threshold)/(label.shape[1]*label.shape[2]))
        feature = feature[indices]
        label = label[indices]
    elif filter_method == 'all':
        # retain everything
        pass
    else:
        raise ValueError("Unknown filter: %s" % filter_method)

    # merge all non-zero labels (to get outline of "complete" tumor)
    label = (label > 0)

    return feature.astype('int'), label.astype('uint8')


def find_files(path,
               group,
               modality,
               extension='.mha',
               ground_truth_modality='OT'):
    """
    Find files with specified extension in specified path
    matching specified group (top-level dir) and modality.
    Returns a list of tuples (feature_filename, ground_truth_filename)
    """
    if group:
        # only look from files in specified group
        path = os.path.join(path, group)

    files = []
    for dirpath, dirnames, filenames in os.walk(path, followlinks=True):
        for filename in filenames:
            # look for .mha files matching the specified modality
            if filename.endswith(extension):
                if not modality or modality in filename:
                    filename = os.path.join(dirpath, filename)
                    if modality != ground_truth_modality:
                        # now look for ground truth
                        ground_truth = find_files(os.path.dirname(dirpath),
                                                  None,
                                                  ground_truth_modality,
                                                  extension)
                        if len(ground_truth) != 1:
                            raise ValueError("Expected 1 ground-truth for %s, found %d"
                                             % (filename, len(ground_truth)))
                        files.append((filename, ground_truth[0]))
                    else:
                        files.append(filename)

    return files
