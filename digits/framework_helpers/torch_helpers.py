# Copyright (c) 2016, NVIDIA CORPORATION.  All rights reserved.
import os
import sys
import subprocess
import tempfile
import PIL

import digits
from digits.config import load_config, config_value
load_config()
from digits import utils, log

def save_max_activations(network_path,weights_path,height,width,layer,units=[-1],mean_file_path=None,gpu=None,logger=None):
    if config_value('torch_root') == '<PATHS>':
        torch_bin = 'th'
    else:
        torch_bin = os.path.join(config_value('torch_root'), 'bin', 'th')

    args = [torch_bin,
            os.path.join(os.path.dirname(os.path.dirname(digits.__file__)),'tools','torch','wrapper.lua'),
            'gradientOptimizer.lua',
            '--network=%s' % os.path.basename(network_path).split(".")[0],
            '--weights=%s' % os.path.split(weights_path)[1],
            '--networkDirectory=%s' % os.path.split(network_path)[0],
            '--height=%s' % height,
            '--width=%s' % width,
            '--chain=%s' % layer,
            '--units=%s' % (','.join(str(x) for x in units))
            ]

    # Convert them all to strings
    args = [str(x) for x in args]

    env = os.environ.copy()

    if mean_file_path is not None:
        args.append('--mean_file_path=%s' % mean_file_path)

    if gpu is not None:
        args.append('--type=cuda')
        # make only the selected GPU visible
        env['CUDA_VISIBLE_DEVICES'] = "%d" % gpu
    else:
        args.append('--type=float')

    # Append units at end:
    p = subprocess.Popen(args,
            cwd=os.path.split(network_path)[0],
            close_fds=True,
            env=env
            )
    p.wait()

def save_weights(network_path,weights_path, gpu=None, logger=None):

    if config_value('torch_root') == '<PATHS>':
        torch_bin = 'th'
    else:
        torch_bin = os.path.join(config_value('torch_root'), 'bin', 'th')

    args = [torch_bin,
            os.path.join(os.path.dirname(os.path.dirname(digits.__file__)),'tools','torch','wrapper.lua'),
            'getWeights.lua',
            '--network=%s' % os.path.basename(network_path).split(".")[0],
            '--networkDirectory=%s' % os.path.split(network_path)[0],
            '--snapshot=%s' % os.path.split(weights_path)[1],
            '--save=%s' % ".",
             '--type=%s' % "float"
            ]

    # Convert them all to strings
    args = [str(x) for x in args]

    env = os.environ.copy()

    p = subprocess.Popen(args,
            cwd=os.path.split(network_path)[0],
            close_fds=True,
            env=env
            )
    p.wait()

def save_activations_and_weights(image_path,network_path,weights_path,image_info=None,labels_dir=None,gpu=None, logger=None):
    if image_info is None:
        image_info = {"height": 256, "width": 256, "channels": 3, "resize_mode": "squash"}

    if config_value('torch_root') == '<PATHS>':
        torch_bin = 'th'
    else:
        torch_bin = os.path.join(config_value('torch_root'), 'bin', 'th')

    # Resize image
    image = utils.image.load_image(image_path)
    image = utils.image.resize_image(image,
                image_info["height"], image_info["width"],
                channels    = image_info["channels"],
                resize_mode = image_info["resize_mode"],
                )

    # Save resized image temporarily
    temp_image_handle, temp_image_path = tempfile.mkstemp(suffix='.png')
    os.close(temp_image_handle)
    image = PIL.Image.fromarray(image)
    try:
        image.save(temp_image_path, format='png')
    except KeyError:
        error_message = 'Unable to save file to "%s"' % temp_image_path
        if logger:
            logger.error(error_message)
        raise digits.inference.errors.InferenceError(error_message)

    args = [torch_bin,
            os.path.join(os.path.dirname(os.path.dirname(digits.__file__)),'tools','torch','wrapper.lua'),
            'test.lua',
            '--image=%s' % temp_image_path,
            '--network=%s' % os.path.basename(network_path).split(".")[0],
            '--networkDirectory=%s' % os.path.split(network_path)[0],
            '--snapshot=%s' % os.path.split(weights_path)[1],
            '--allPredictions=yes',
            '--visualization=yes',
            '--save=%s' % "."
            ]

    # Convert them all to strings
    args = [str(x) for x in args]

    env = os.environ.copy()

    if gpu is not None:
        args.append('--type=cuda')
        # make only the selected GPU visible
        env['CUDA_VISIBLE_DEVICES'] = "%d" % gpu
    else:
        args.append('--type=float')

    # TODO: Get mean
    args.append('--subtractMean=none')

    p = subprocess.Popen(args,
            cwd=os.path.split(network_path)[0],
            close_fds=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            env=env
            )
    p.wait()
