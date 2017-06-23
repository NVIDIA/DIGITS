# Copyright (c) 2014-2017, NVIDIA CORPORATION.  All rights reserved.
from __future__ import absolute_import

from .caffe_train import CaffeTrainTask, CaffeTrainSanityCheckError

from google.protobuf import text_format
from digits import test_utils

# Must import after importing digit.config
import caffe_pb2


def check_positive(desc, stage):
    network = caffe_pb2.NetParameter()
    text_format.Merge(desc, network)
    CaffeTrainTask.net_sanity_check(network, stage)


def check_negative(desc, stage):
    network = caffe_pb2.NetParameter()
    text_format.Merge(desc, network)
    try:
        CaffeTrainTask.net_sanity_check(network, stage)
    except CaffeTrainSanityCheckError:
        pass


class TestCaffeNetSanityCheck(test_utils.CaffeMixin):

    # positive cases

    def test_std_net_train(self):
        desc = \
            """
            layer {
                name: "data"
                type: "Data"
                top: "data"
                top: "label"
                include {
                    phase: TRAIN
                }
            }
            layer {
                name: "hidden"
                type: 'InnerProduct'
                bottom: "data"
                top: "output"
            }
            layer {
                name: "loss"
                type: "SoftmaxWithLoss"
                bottom: "output"
                bottom: "label"
                top: "loss"
            }
            layer {
                name: "accuracy"
                type: "Accuracy"
                bottom: "output"
                bottom: "label"
                top: "accuracy"
                include {
                    phase: TEST
                }
            }
            """
        check_positive(desc, caffe_pb2.TRAIN)

    def test_std_net_deploy(self):
        desc = \
            """
            input: "data"
            layer {
                name: "hidden"
                type: 'InnerProduct'
                bottom: "data"
                top: "output"
            }
            """
        check_positive(desc, caffe_pb2.TEST)

    def test_ref_label_with_proper_include_directive(self):
        desc = \
            """
            input: "data"
            layer {
                name: "hidden"
                type: 'InnerProduct'
                bottom: "data"
                top: "output"
            }
            layer {
                name: "loss"
                type: "SoftmaxWithLoss"
                bottom: "output"
                bottom: "label"
                top: "loss"
                include {
                    phase: TRAIN
                }
            }
            """
        check_positive(desc, caffe_pb2.TEST)

    def test_ref_label_with_proper_exclude_directive(self):
        desc = \
            """
            input: "data"
            layer {
                name: "hidden"
                type: 'InnerProduct'
                bottom: "data"
                top: "output"
            }
            layer {
                name: "lossExcludedInTest"
                type: "SoftmaxWithLoss"
                bottom: "output"
                bottom: "label"
                top: "loss"
                exclude {
                    phase: TEST
                }
            }
            """
        check_positive(desc, caffe_pb2.TEST)

    # negative cases

    def test_error_ref_label_in_deploy(self):
        desc = \
            """
            input: "data"
            layer {
                name: "hidden"
                type: 'InnerProduct'
                bottom: "data"
                top: "output"
            }
            layer {
                name: "loss"
                type: "SoftmaxWithLoss"
                bottom: "output"
                bottom: "label"
                top: "loss"
            }
            """
        check_negative(desc, caffe_pb2.TEST)

    def test_error_ref_unknown_blob(self):
        desc = \
            """
            input: "data"
            layer {
                name: "hidden"
                type: 'InnerProduct'
                bottom: "data"
                bottom: "bogusBlob"
                top: "output"
            }
            """
        check_negative(desc, caffe_pb2.TRAIN)

    def test_error_ref_unincluded_blob(self):
        desc = \
            """
            input: "data"
            layer {
                name: "hidden"
                type: 'InnerProduct'
                bottom: "data"
                top: "output"
                include {
                    phase: TRAIN
                }
            }
            layer {
                name: "hidden"
                type: 'InnerProduct2'
                bottom: "data"
                top: "output"
            }
            layer {
                name: "loss"
                type: "SoftmaxWithLoss"
                bottom: "output"
                bottom: "label"
                top: "loss"
                include {
                    phase: TRAIN
                }
            }
            """
        check_negative(desc, caffe_pb2.TEST)

    def test_error_ref_excluded_blob(self):
        desc = \
            """
            input: "data"
            layer {
                name: "hidden"
                type: 'InnerProduct'
                bottom: "data"
                top: "output"
                include {
                    phase: TRAIN
                }
            }
            layer {
                name: "hidden"
                type: 'InnerProduct2'
                bottom: "data"
                top: "output"
            }
            layer {
                name: "loss"
                type: "SoftmaxWithLoss"
                bottom: "output"
                bottom: "label"
                top: "loss"
                exclude {
                    phase: TEST
                }
            }
            """
        check_negative(desc, caffe_pb2.TEST)
