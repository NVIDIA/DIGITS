#!/usr/bin/env python2
# Copyright (c) 2014-2017, NVIDIA CORPORATION.  All rights reserved.

import argparse
import logging
import os
import random
import requests
import re
import sys
import time
import urllib

# Add path for DIGITS package
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import digits.config  # noqa
from digits import utils, log  # noqa

logger = logging.getLogger('digits.tools.parse_folder')


def unescape(s):
    return urllib.unquote(s)


def validate_folder(folder):
    if utils.is_url(folder):
        try:
            r = requests.head(folder, timeout=utils.HTTP_TIMEOUT)
            if r.status_code not in [requests.codes.ok, requests.codes.moved, requests.codes.found]:
                logger.error('"%s" returned status_code %s' % (folder, r.status_code))
                return False
        except Exception as e:
            logger.error('%s: %s' % (type(e).__name__, e))
            return False
        return True
    if not os.path.exists(folder):
        logger.error('folder "%s" does not exist' % folder)
        return False
    if not os.path.isdir(folder):
        logger.error('"%s" is not a directory' % folder)
        return False
    if not os.access(folder, os.R_OK):
        logger.error('you do not have read access to folder "%s"' % folder)
        return False
    return True


def validate_output_file(filename):
    if filename is None:
        return True
    if os.path.exists(filename):
        logger.error('output file "%s" already exists!' % filename)
        return False
    output_dir = os.path.dirname(filename)
    if not output_dir:
        output_dir = '.'
    if not os.path.exists(output_dir):
        logger.error('output directory "%s" does not exist!' % output_dir)
        return False
    if not os.access(output_dir, os.W_OK):
        logger.error('you do not have write access to output directory "%s"!' % output_dir)
        return False
    return True


def validate_input_file(filename):
    if not os.path.exists(filename) or not os.path.isfile(filename):
        logger.error('input file "%s" does not exist!' % filename)
        return False
    if not os.access(filename, os.R_OK):
        logger.error('you do not have read access to "%s"!' % filename)
        return False
    return True


def validate_range(number, min_value=None, max_value=None, allow_none=False):
    if number is None:
        if allow_none:
            return True
        else:
            logger.error('invalid value %s' % number)
            return False
    try:
        float(number)
    except ValueError:
        logger.error('invalid value %s' % number)
        return False

    if min_value is not None and number < min_value:
        logger.error('invalid value %s' % number)
        return False
    if max_value is not None and number > max_value:
        logger.error('invalid value %s' % number)
        return False
    return True


def calculate_percentages(labels_file,
                          train_file, percent_train,
                          val_file, percent_val,
                          test_file, percent_test,
                          **kwargs):
    """
    Returns (percent_train, percent_val, percent_test)
    Throws exception on errors
    """
    # reject any percentages not between 0-100
    assert all(x is None or 0 <= x <= 100
               for x in [percent_train, percent_val, percent_test]), \
        'all percentages must be 0-100 inclusive or not specified'

    # return values
    pt = None
    pv = None
    ps = None
    # making these sets
    mt = False
    mv = False
    ms = False

    if train_file is not None:
        pt = percent_train
        mt = True
    if val_file is not None:
        pv = percent_val
        mv = True
    if test_file is not None:
        ps = percent_test
        ms = True

    making = sum([mt, mv, ms])
    assert making > 0, 'must specify at least one of train_file, val_file and test_file'
    if train_file is not None:
        assert validate_output_file(labels_file)
    else:
        assert validate_input_file(labels_file)

    if making == 1:
        if mt:
            return (100, 0, 0)
        elif mv:
            return (0, 100, 0)
        else:
            return (0, 0, 100)
    elif making == 2:
        if mt and mv:
            assert not (pt is None and pv is None), 'must give percent_train or percent_val'
            if pt is not None and pv is not None:
                assert (pt + pv) == 100, 'percentages do not sum to 100'
                return (pt, pv, 0)
            elif pt is not None:
                return (pt, 100 - pt, 0)
            else:
                return (100 - pv, pv, 0)
        elif mt and ms:
            assert not (pt is None and ps is None), 'must give percent_train or percent_test'
            if pt is not None and ps is not None:
                assert (pt + ps) == 100, 'percentages do not sum to 100'
                return (pt, 0, ps)
            elif pt is not None:
                return (pt, 0, 100 - pt)
            else:
                return (100 - ps, 0, ps)
        elif mv and ms:
            assert not (pv is None and ps is None), 'must give percent_val or percent_test'
            if pv is not None and ps is not None:
                assert (pv + ps) == 100, 'percentages do not sum to 100'
                return (0, pv, ps)
            elif pv is not None:
                return (0, pv, 100 - pv)
            else:
                return (0, 100 - ps, ps)
    elif making == 3:
        specified = sum([pt is not None, pv is not None, ps is not None])
        assert specified >= 2, 'must specify two of percent_train, percent_val, and percent_test'
        if specified == 3:
            assert (pt + pv + ps) == 100, 'percentages do not sum to 100'
            return (pt, pv, ps)
        elif specified == 2:
            if pt is None:
                assert (pv + ps) <= 100, 'percentages cannot exceed 100'
                return (100 - (pv + ps), pv, ps)
            elif pv is None:
                assert (pt + ps) <= 100, 'percentages cannot exceed 100'
                return (pt, 100 - (pt + ps), ps)
            elif ps is None:
                assert (pt + pv) <= 100, 'percentages cannot exceed 100'
                return (pt, pv, 100 - (pt + pv))


def parse_web_listing(url):
    """Utility for parse_folder()

    Parses an autoindexed folder into directories and files
    Returns (dirs, files)
    """
    dirs = []
    files = []

    r = requests.get(url, timeout=3.05)
    if r.status_code != requests.codes.ok:
        raise Exception('HTTP Status Code %s' % r.status_code)

    for line in r.content.split('\n'):
        line = line.strip()
        # Matches nginx and apache's autoindex formats
        match = re.match(
            r'^.*\<a.+href\=[\'\"]([^\'\"]+)[\'\"].*\>.*(\w{1,4}-\w{1,4}-\w{1,4})', line, flags=re.IGNORECASE)
        if match:
            if match.group(1).endswith('/'):
                dirs.append(match.group(1))

            elif match.group(1).lower().endswith(utils.image.SUPPORTED_EXTENSIONS):
                files.append(match.group(1))
    return (dirs, files)


def web_listing_all_files(url, count=0, max_count=None):
    """Utility for parse_folder()
    Gets all files from a url by parsing the directory and all subdirectories looking for image files
    Returns (urls, count)
    (recursive)
    """
    urls = []
    dirs, files = parse_web_listing(url)
    for f in files:
        urls.append(url + f)
        count += 1
        if max_count is not None and count >= max_count:
            logger.warning('Reached maximum limit for this category')
            return urls, count
    for d in dirs:
        new_urls, count = web_listing_all_files(url + d, count, max_count)
        urls += new_urls
        if max_count is not None and count >= max_count:
            break
    return urls, count


def three_way_split_indices(size, pct_b, pct_c):
    """
    Utility for splitting an array
    Returns (a, b) where a and b are indices for splitting the array into 3 pieces

    Arguments:
    size -- the size of the array
    pct_b -- the percent of the array that should be used for group b
    pct_c -- the percent of the array that should be used for group c
    """
    assert 0 <= pct_b <= 100
    assert 0 <= pct_c <= 100
    pct_a = 100 - (pct_b + pct_c)
    assert 0 <= pct_a <= 100

    if pct_a >= 100:
        return size, size
    elif pct_b >= 100:
        return 0, size
    elif pct_c >= 100:
        return 0, 0
    else:
        a = int(round(float(size) * pct_a / 100))
        if pct_a and not a:
            a = 1
        b = int(round(float(size) * pct_b / 100))
        if a + b > size:
            b = size - a
        if pct_b and not b:
            if a > 1:
                a -= 1
                b = 1
            elif a != size:
                b = 1
        c = size - (a + b)
        if pct_c and not c:
            if b > 1:
                b -= 1
                c = 1
            elif a > 1:
                a -= 1
                c = 1
        assert a + b + c == size
        return a, a + b


def parse_folder(folder, labels_file,
                 train_file=None, percent_train=None,
                 val_file=None, percent_val=None,
                 test_file=None, percent_test=None,
                 min_per_category=2,
                 max_per_category=None,
                 ):
    """
    Parses a folder of images into three textfiles
    Returns True on success

    Arguments:
    folder -- a folder containing folders of images (can be a filesystem path or a url)
    labels_file -- file for labels

    Keyword Arguments:
    train_file -- output file for training images
    percent_test -- percentage of images to use in the training set
    val_file -- output file for validation images
    percent_val -- percentage of images to use in the validation set
    test_file -- output file for test images
    percent_test -- percentage of images to use in the test set
    min_per_category -- minimum number of images per category
    max_per_category -- maximum number of images per category
    """
    create_labels = (percent_train > 0)
    labels = []

    # Read the labels from labels_file

    if not create_labels:
        with open(labels_file) as infile:
            for line in infile:
                line = line.strip()
                if line:
                    labels.append(line)

    # Verify that at least two category folders exist

    folder_is_url = utils.is_url(folder)
    if folder_is_url:
        if not folder.endswith('/'):
            folder += '/'
        subdirs, _ = parse_web_listing(folder)
    else:
        if os.path.exists(folder) and os.path.isdir(folder):
            subdirs = []
            for filename in os.listdir(folder):
                subdir = os.path.join(folder, filename)
                if os.path.isdir(subdir):
                    subdirs.append(subdir)
        else:
            logger.error('folder does not exist')
            return False

    subdirs.sort()

    if len(subdirs) < 2:
        logger.error('folder must contain at least two subdirectories')
        return False

    # Parse the folder

    train_count = 0
    val_count = 0
    test_count = 0

    if percent_train:
        train_outfile = open(train_file, 'w')
    if percent_val:
        val_outfile = open(val_file, 'w')
    if percent_test:
        test_outfile = open(test_file, 'w')

    subdir_index = 0
    label_index = 0
    for subdir in subdirs:
        # Use the directory name as the label
        label_name = subdir
        if folder_is_url:
            label_name = unescape(label_name)
        else:
            label_name = os.path.basename(label_name)
        label_name = label_name.replace('_', ' ')
        if label_name.endswith('/'):
            # Remove trailing slash
            label_name = label_name[0:-1]

        if create_labels:
            labels.append(label_name)
            label_index = len(labels) - 1
        else:
            found = False
            for i, l in enumerate(labels):
                if label_name == l:
                    found = True
                    label_index = i
                    break
            if not found:
                logger.warning('Category "%s" not found in labels_file. Skipping.' % label_name)
                continue

        logger.debug('Category - %s' % label_name)

        lines = []

        # Read all images in the folder

        if folder_is_url:
            urls, _ = web_listing_all_files(folder + subdir, max_count=max_per_category)
            for url in urls:
                lines.append('%s %d' % (url, label_index))
        else:
            for dirpath, dirnames, filenames in os.walk(os.path.join(folder, subdir), followlinks=True):
                for filename in filenames:
                    if filename.lower().endswith(utils.image.SUPPORTED_EXTENSIONS):
                        lines.append('%s %d' % (os.path.join(folder, subdir, dirpath, filename), label_index))
                        if max_per_category is not None and len(lines) >= max_per_category:
                            break
                if max_per_category is not None and len(lines) >= max_per_category:
                    logger.warning('Reached maximum limit for this category')
                    break

        # Split up the lines

        train_lines = []
        val_lines = []
        test_lines = []

        required_categories = 0
        if percent_train > 0:
            required_categories += 1
        if percent_val > 0:
            required_categories += 1
        if percent_test > 0:
            required_categories += 1

        if not lines or len(lines) < required_categories or len(lines) < min_per_category:
            logger.warning('Not enough images for this category')
            labels.pop()
        else:
            random.shuffle(lines)
            a, b = three_way_split_indices(len(lines), percent_val, percent_test)
            train_lines = lines[:a]
            val_lines = lines[a:b]
            test_lines = lines[b:]

        if train_lines:
            train_outfile.write('\n'.join(train_lines) + '\n')
            train_count += len(train_lines)
        if val_lines:
            val_outfile.write('\n'.join(val_lines) + '\n')
            val_count += len(val_lines)
        if test_lines:
            test_outfile.write('\n'.join(test_lines) + '\n')
            test_count += len(test_lines)

        subdir_index += 1
        logger.debug('Progress: %0.2f' % (float(subdir_index) / len(subdirs)))

    if percent_train:
        train_outfile.close()
    if percent_val:
        val_outfile.close()
    if percent_test:
        test_outfile.close()

    if create_labels:
        if len(labels) < 2:
            logger.error('Did not find two valid categories')
            return False
        else:
            with open(labels_file, 'w') as labels_outfile:
                labels_outfile.write('\n'.join(labels) + '\n')

    logger.info('Found %d images in %d categories.' % (train_count + val_count + test_count, len(labels)))
    logger.info('Selected %d for training.' % train_count)
    logger.info('Selected %d for validation.' % val_count)
    logger.info('Selected %d for testing.' % test_count)
    return True


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Parse-Folder tool - DIGITS')

    # Positional arguments

    parser.add_argument(
        'folder',
        help='A filesystem path or url to the folder of images'
    )
    parser.add_argument(
        'labels_file',
        help=('The file containing labels. If train_file is set, this file '
              'will be generated (output). Otherwise, this file will be read (input).')
    )

    # Optional arguments

    parser.add_argument(
        '-t', '--train_file',
        help='The output file for training images'
    )
    parser.add_argument(
        '-T', '--percent_train', type=float,
        help='Percent of images used for the training set (constant across all categories)'
    )
    parser.add_argument(
        '-v', '--val_file',
        help='The output file for validation images'
    )
    parser.add_argument(
        '-V', '--percent_val', type=float,
        help='Percent of images used for the validation set (constant across all categories)'
    )
    parser.add_argument(
        '-s', '--test_file',
        help='The output file for test images'
    )
    parser.add_argument(
        '-S', '--percent_test', type=float,
        help='Percent of images used for the test set (constant across all categories)'
    )
    parser.add_argument(
        '--min', type=int, metavar='MIN_PER_CATEGORY', default=1,
        help=("What is the minimum allowable number of images per category? "
              "(categories which don't meet this criteria will be ignored) [default=2]")
    )
    parser.add_argument(
        '--max', type=int, metavar='MAX_PER_CATEGORY',
        help=("What is the maximum limit of images per category? "
              "(categories which exceed this limit will be trimmed down) [default=None]")
    )

    args = vars(parser.parse_args())

    for valid in [
            validate_folder(args['folder']),
            validate_range(args['percent_train'],
                           min_value=0, max_value=100, allow_none=True),
            validate_output_file(args['train_file']),
            validate_range(args['percent_val'],
                           min_value=0, max_value=100, allow_none=True),
            validate_output_file(args['val_file']),
            validate_range(args['percent_test'],
                           min_value=0, max_value=100, allow_none=True),
            validate_output_file(args['test_file']),
            validate_range(args['min'], min_value=1),
            validate_range(args['max'], min_value=1, allow_none=True),
    ]:
        if not valid:
            sys.exit(1)

    try:
        percent_train, percent_val, percent_test = calculate_percentages(**args)
    except Exception as e:
        logger.error('%s: %s' % (type(e).__name__, e))
        sys.exit(1)

    start_time = time.time()

    if parse_folder(args['folder'], args['labels_file'],
                    train_file=args['train_file'],
                    percent_train=percent_train,
                    val_file=args['val_file'],
                    percent_val=percent_val,
                    test_file=args['test_file'],
                    percent_test=percent_test,
                    min_per_category=args['min'],
                    max_per_category=args['max'],
                    ):
        logger.info('Done after %d seconds.' % (time.time() - start_time))
        sys.exit(0)
    else:
        sys.exit(1)
