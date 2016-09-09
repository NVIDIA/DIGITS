#!/usr/bin/env python2
# Copyright (c) 2016, NVIDIA CORPORATION.  All rights reserved.
import argparse
import json
import logging
import os
import sys

import h5py
import numpy as np
from google.protobuf import text_format
from google.protobuf.descriptor import FieldDescriptor as FD

# Add path for DIGITS package
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import digits.config
digits.config.load_config()

from digits import utils, log
from digits.inference.errors import InferenceError

# must call digits.config.load_config() before caffe to set the path
import caffe

logger = logging.getLogger('digits.tools.inference')

def get_weights(output_dir,net):
    """
    Get weights from a pretrained model
    """
    f = h5py.File(os.path.join(output_dir,'filters.hdf5'),'w')

    # Save param keys to file:
    num_outputs = len(net.params)
    for index, layer in enumerate(net.params):

        shape = net.params[layer][0].data.shape
        raw_data = np.reshape(np.array(net.params[layer][0].data),shape)

        vis_data = utils.image.reshape_data_for_vis(raw_data,'BGR')
        dset = f.create_dataset(layer, data=utils.image.normalize_data(vis_data))

        # TODO: Add more stats
        dset.attrs['stats'] = json.dumps({"shape": shape, "num_activations": shape[0]})

        logger.info('Processed %s/%s', index, num_outputs)

    f.close()

def pb2json(pb):
    """
    From https://github.com/vadimkantorov/caffemodel2json
    Copyright (c) 2015 Vadim Kantorov, The MIT License (MIT)
    """
    _ftype2js = {
        FD.TYPE_DOUBLE: float,
        FD.TYPE_FLOAT: float,
        FD.TYPE_INT64: long,
        FD.TYPE_UINT64: long,
        FD.TYPE_INT32: int,
        FD.TYPE_FIXED64: float,
        FD.TYPE_FIXED32: float,
        FD.TYPE_BOOL: bool,
        FD.TYPE_STRING: unicode,
        FD.TYPE_BYTES: lambda x: x.encode('string_escape'),
        FD.TYPE_UINT32: int,
        FD.TYPE_ENUM: int,
        FD.TYPE_SFIXED32: float,
        FD.TYPE_SFIXED64: float,
        FD.TYPE_SINT32: int,
        FD.TYPE_SINT64: long,
    }
    js = {}
    fields = pb.ListFields()	#only filled (including extensions)
    for field,value in fields:
        if field.type == FD.TYPE_MESSAGE:
            ftype = pb2json
        elif field.type in _ftype2js:
            ftype = _ftype2js[field.type]

        if field.label == FD.LABEL_REPEATED:
            if field.name is "data":
                js_value = len(value)
            else:
                js_value = []
                for v in value:
                    js_value.append(ftype(v))
        else:
            js_value = ftype(value)
        js[field.name] = js_value
    return js

def run(output_dir, model_def_path, weights_path):

    # net = caffe.proto.caffe_pb2.NetParameter()
    # with open(weights_path, 'rb') as infile:
    #    net.MergeFromString(infile.read())
    caffe.set_mode_cpu()
    net = caffe.Net(model_def_path,weights_path,caffe.TEST)
    get_weights(output_dir, net)

    # Get Network Parameter from Prototxt
    deploy_network = caffe.proto.caffe_pb2.NetParameter()
    with open(model_def_path, 'r') as f:
       model_def = f.read()
    text_format.Merge(model_def, deploy_network)

    # Write to JSON:
    model_def_json = pb2json(deploy_network)
    output_file = os.path.join(output_dir, "model_def.json")
    with open(output_file, 'w') as outfile:
        json.dump(model_def_json, outfile,indent=4, separators=(',', ': '))


    logger.info('Saved data to %s', output_dir)

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Weights tool for pretrained models - DIGITS')

    ### Positional arguments
    parser.add_argument('output_dir',
            help='Directory to write outputs to')

    parser.add_argument('model_def_path',
            help='Path to model definition',
            )

    parser.add_argument('weights_path',
            help='Path to weights',
            )

    args = vars(parser.parse_args())

    try:
        run(
            args['output_dir'],
            args['model_def_path'],
            args['weights_path']
            )
    except Exception as e:
        logger.error('%s: %s' % (type(e).__name__, e.message))
        raise
