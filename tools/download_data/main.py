#!/usr/bin/env python
# Copyright (c) 2015, NVIDIA CORPORATION.  All rights reserved.

import sys
import time
import argparse

from mnist import MnistDownloader
from cifar10 import Cifar10Downloader
from cifar100 import Cifar100Downloader

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Download-Data tool - DIGITS')

    ### Positional arguments

    parser.add_argument('dataset',
            help='mnist/cifar10/cifar100'
            )
    parser.add_argument('output_dir',
            help='The output directory for the data'
            )

    ### Optional arguments

    parser.add_argument('-c', '--clean',
            action = 'store_true',
            help='clean out the directory first (if it exists)'
            )

    args = vars(parser.parse_args())

    dataset = args['dataset'].lower()

    start = time.time()
    if dataset == 'mnist':
        d = MnistDownloader(
                outdir  = args['output_dir'],
                clean   = args['clean'])
        d.getData()
    elif dataset == 'cifar10':
        d = Cifar10Downloader(
                outdir  = args['output_dir'],
                clean   = args['clean'])
        d.getData()
    elif dataset == 'cifar100':
        d = Cifar100Downloader(
                outdir  = args['output_dir'],
                clean   = args['clean'])
        d.getData()
    else:
        print 'Unknown dataset "%s"' % args['dataset']
        sys.exit(1)

    print 'Done after %s seconds.' % (time.time() - start)

