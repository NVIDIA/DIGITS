#!/usr/bin/env python2
# Copyright (c) 2016-2017, NVIDIA CORPORATION.  All rights reserved.
"""
Downloads BVLC Alexnet and perform the require net surgery to convert into an FCN Alexnet
"""

import urllib

import caffe

ALEXNET_PROTOTXT_URL = "https://raw.githubusercontent.com/BVLC/caffe/rc3/models/bvlc_alexnet/deploy.prototxt"
ALEXNET_PROTOTXT_FILENAME = "bvlc_alexnet.deploy.prototxt"
ALEXNET_MODEL_URL = "http://dl.caffe.berkeleyvision.org/bvlc_alexnet.caffemodel"
ALEXNET_MODEL_FILENAME = "bvlc_alexnet.caffemodel"

FCN_ALEXNET_PROTOTXT_FILENAME = "fcn_alexnet.deploy.prototxt"
FCN_ALEXNET_MODEL_FILENAME = "fcn_alexnet.caffemodel"


def download(url, filename):
    print "Downloading %s..." % url
    urllib.urlretrieve(url, filename)


def generate_fcn():
    # download files
    print "Downloading files (this might take a few minutes)..."
    download(ALEXNET_PROTOTXT_URL, ALEXNET_PROTOTXT_FILENAME)
    download(ALEXNET_MODEL_URL, ALEXNET_MODEL_FILENAME)

    caffe.set_mode_cpu()

    print "Loading Alexnet model..."
    alexnet = caffe.Net(ALEXNET_PROTOTXT_FILENAME, ALEXNET_MODEL_FILENAME, caffe.TEST)

    print "Loading FCN-Alexnet prototxt..."
    fcn_alexnet = caffe.Net(FCN_ALEXNET_PROTOTXT_FILENAME, caffe.TEST)

    print "Transplanting parameters..."
    transplant(fcn_alexnet, alexnet)

    print "Saving FCN-Alexnet model to %s " % FCN_ALEXNET_MODEL_FILENAME
    fcn_alexnet.save(FCN_ALEXNET_MODEL_FILENAME)


def transplant(new_net, net, suffix=''):
    # from fcn.berkeleyvision.org
    for p in net.params:
        p_new = p + suffix
        if p_new not in new_net.params:
            print 'dropping', p
            continue
        for i in range(len(net.params[p])):
            if i > (len(new_net.params[p_new]) - 1):
                print 'dropping', p, i
                break
            if net.params[p][i].data.shape != new_net.params[p_new][i].data.shape:
                print 'coercing', p, i, 'from', net.params[p][i].data.shape, 'to', new_net.params[p_new][i].data.shape
            else:
                print 'copying', p, ' -> ', p_new, i
            new_net.params[p_new][i].data.flat = net.params[p][i].data.flat


if __name__ == '__main__':
    generate_fcn()
