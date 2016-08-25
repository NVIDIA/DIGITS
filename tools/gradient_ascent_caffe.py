#!/usr/bin/env python2
# Copyright (c) 2016, NVIDIA CORPORATION.  All rights reserved.

import argparse
import h5py
import json
import logging
import numpy as np
import os
import sys
import cv2
from scipy.optimize import curve_fit
from scipy import exp
from google.protobuf import text_format

# Add path for DIGITS package
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import digits.config
digits.config.load_config()

from digits import utils, log
from digits.inference.errors import InferenceError

from tools.gradient_ascent.gradient_optimizer import GradientOptimizer, FindParams
# must call digits.config.load_config() before caffe to set the path
import caffe
from caffe.proto import caffe_pb2

logger = logging.getLogger('digits.tools.inference')

"""
Get max activations from a pretrained model
"""
def get_mean(mean_file_path, data_size):
    # Get mean from binaryprotot
    mean_blob = caffe.proto.caffe_pb2.BlobProto()
    mean_proto = open( mean_file_path , 'rb' ).read()
    mean_blob.ParseFromString(mean_proto)

    # Convert Mean Blob for Resizing with cv2 to input dimensions
    mean_image = np.transpose(np.array( caffe.io.blobproto_to_array(mean_blob) )[0], (1,2,0))
    im = cv2.resize(mean_image,data_size)
    if len(im.shape) == 2:
        # cv2 removed color channel, so re-add it:
        data_mean = np.array([im])
    else:
        data_mean = np.transpose(cv2.resize(mean_image,data_size), (2,1,0))

    return data_mean

def infer(output_dir,model_def_path, weights_path, layer,units, mean_file_path=None, gpu=None):
    # Set Mode to Run Inference:
    if gpu is not None:
        caffe.set_device(gpu)
        caffe.set_mode_gpu()
    else:
        caffe.set_mode_cpu()

    # Update Model Def to allow force backward
    deploy_network = caffe.proto.caffe_pb2.NetParameter()
    with open(model_def_path, 'r') as f:
       model_def = f.read()
    text_format.Merge(model_def, deploy_network)
    text_format.Merge("force_backward: true", deploy_network)
    with open(model_def_path, 'w') as outfile:
       text_format.PrintMessage(deploy_network, outfile)

    # TODO : Make channel swap inputable
    net = caffe.Classifier(
        model_def_path,
        weights_path
    )

    # Get the input shape in order to resize mean image:
    in_ = net.inputs[0]
    input_shape = net.blobs[in_].data.shape

    data_size = (input_shape[2], input_shape[3])

    if mean_file_path is not None:
        # If mean entered, use is:
        mean = get_mean(mean_file_path, data_size)
    else:
        # Else generate grey image:
        mean = np.ones(net.blobs[in_].data[0].shape) * 150

    # For now , just use grey image:
    mean = np.ones(net.blobs[in_].data[0].shape) * 255

    # Set the mean for the network (as it wasnt set during initialization)
    transformer = caffe.io.Transformer({in_: input_shape})
    transformer.set_mean(in_, mean)
    # cv2.imshow('gradient',np.transpose(net.blobs[in_].data[0], (1,2,0)));
    # cv2.waitKey(0);

    if layer in net.blobs:
        out = net.blobs[layer].data
    else:
        logger.info('Error: %s', "Layer Not Optimizable")
        sys.exit()

    # Check if fully convolutional layer, or a convolutional layer
    # If convolutional set spacial to be the center to avoid cropping

    is_conv = (len(out.shape) == 4)
    if (is_conv):
        min_spatial = min((out.shape[2], out.shape[3]))
        push_spatial = (min_spatial/2, min_spatial/2)
    else:
        push_spatial = (0,0)

    optimizer = GradientOptimizer(net,mean)

    if -1 in units:
        units = range(out.shape[1])

    for i, unit in enumerate(units):
        # TODO: Make the params below optionable:
        params = FindParams(
            push_layer = layer,
            push_channel = unit,
            decay = 0.0001,
            blur_radius = 1.0,
            blur_every = 4,
            max_iter = 200,
            push_spatial = push_spatial,
            lr_params = {'lr': 100.0}
        )

        try:
            im = optimizer.run_optimize(params, prefix_template = "blah",brave = True,save=False)
            # cv2.imshow('gradient',im);
            # cv2.waitKey(0);
        except:
            logger.info('Error: %s', "Optimization Failure")
            sys.exit()

        # im = np.square(np.gradient(np.mean(np.mean(im, axis=2),axis=1)))

        # Get the distribution of points along the image (assuming roughly symmetric)
        y = np.sum(np.square(np.gradient(np.mean(im, axis=2),axis=1)), axis=1)
        x = np.arange(len(y))

        # N, and initial guess for mean, and sigma for bell curve:
        n = len(x)
        mean = np.sum(x*y)/n
        sigma = np.sum(y*(x-mean)**2)/n

        # Define Bell Curve Fit Function:
        def gaus(x,a,x0,sigma):
            return a*exp(-(x-x0)**2/(2*sigma**2))

        # Fit data
        N_max = 10
        fit_success = False
        for j in range(N_max):
            try:
                popt,__ = curve_fit(gaus,x,y,p0=[1,mean,sigma])
                sigma   = int(popt[2])
                if sigma != 0 :
                    fit_success = True
                    break
            except:
                logger.info('Warning: %s', "Could not find receptive field")
                pass

        if fit_success is True and is_conv:
            w = 4*np.abs(sigma)
            y_out = gaus(x,*popt)

            ymax = np.argmax(y_out)
            if (ymax == 1):
               ymax = input_shape[2]/2

            # Plot Gaussian Curve:
            # import matplotlib.pyplot as plt
            # plt.plot(x,y,'b+:',label='data')
            # plt.plot(x,y_out,'ro:',label='fit')
            # plt.legend()
            # plt.xlabel('Pixel')
            # plt.ylabel('Squared Gradient')
            # plt.show()

            # Crop image:
            # TODO: Crop based on width and height (for non-square data layers)
            if (w < ymax):
               cropped = im[ymax-w:ymax+w, ymax-w:ymax+w,:]
            else:
               cropped = im
        else:
            cropped = im

        # Show output
        # cv2.imshow('gradient', cv2.resize(cropped, (input_shape[2],input_shape[3])));
        # cv2.waitKey(0);
        # sys.exit()

        name = "%s/%s" % (layer, unit)

        cropped_name = "%s/cropped" % name
        full_name = "%s/full" % name
        path = os.path.join(output_dir,'max_activations.hdf5')
        with h5py.File(path,'a') as f:
            if cropped_name in f:
                del f[cropped_name]

            if full_name in f:
                del f[full_name]

            f.create_dataset(cropped_name, data=np.transpose(np.uint8(256*cropped),(2,0,1)),dtype='i8')
            # f.create_dataset(full_name, data=np.uint8(256*im),dtype='i8')
            f.close()
        logger.info('Processed %s/%s units', i+1, len(units))


    logger.info('Saved data to %s', output_dir)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Gradient Ascent tool for pretrained models - DIGITS')

    ### Positional arguments
    parser.add_argument('-o', '--output_dir',
            help='Output directory',
            default='.'
            )

    parser.add_argument('-p', '--model_def_path',
            help='Path to model definition.prototxt',
            )

    parser.add_argument('-w', '--weights_path',
            help='Path to weights.caffemodel',
            )

    parser.add_argument('-l', '--layer',
            help='Name of output layer',
            )
    parser.add_argument('-u', '--units',
            type=str,
            default="-1",
            help='Index of units to optimize in output layer',
            )

    ### Optional arguments
    parser.add_argument('-m', '--mean_file_path',
            default=None,
            help='Path to mean.binaryproto',
            )

    parser.add_argument('-g', '--gpu',
            type=int,
            default=None,
            help='GPU to use (as in nvidia-smi output, default: None)',
            )

    args = vars(parser.parse_args())

    try:
        infer(
            args['output_dir'],
            args['model_def_path'],
            args['weights_path'],
            args['layer'],
            map(int,str.split(args['units'],",")),
            args['mean_file_path'],
            args['gpu']
            )
    except Exception as e:
        logger.error('%s: %s' % (type(e).__name__, e.message))
        raise
