# Copyright (c) 2014-2015, NVIDIA CORPORATION.  All rights reserved.

import os.path

from flask import request

from digits import utils
from digits.webapp import app
import classification.views

NAMESPACE = '/datasets/images'

@app.route(NAMESPACE + '/resize-example', methods=['POST'])
def image_dataset_resize_example():
    """Returns a string of png data"""
    try:
        import digits
        example_image_path = os.path.join(os.path.dirname(digits.__file__), 'static', 'images', 'mona_lisa.jpg')
        image = utils.image.load_image(example_image_path)
        if image is None:
            raise Exception('Could not load example image "%s"' % example_image_path)

        width = int(request.form['width'])
        height = int(request.form['height'])
        channels = int(request.form['channels'])
        resize_mode = request.form['resize_mode']

        image = utils.image.resize_image(image, height, width,
                channels=channels,
                resize_mode=resize_mode,
                )
        data = utils.image.embed_image_html(image)
        return '<img src=\"' + data + '\" style=\"width:%spx;height=%spx\" />' % (width, height)
    except Exception as e:
        return '%s: %s' % (type(e).__name__, e)

