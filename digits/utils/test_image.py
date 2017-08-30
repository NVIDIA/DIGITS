# Copyright (c) 2014-2017, NVIDIA CORPORATION.  All rights reserved.
from __future__ import absolute_import

import os
import tempfile

# Find the best implementation available
try:
    from cStringIO import StringIO
except ImportError:
    from StringIO import StringIO

import mock
from nose.tools import assert_raises
import numpy as np
import PIL.Image

from . import errors
from . import image as image_utils
import digits
from digits import test_utils


test_utils.skipIfNotFramework('none')


class TestLoadImage():

    def test_bad_path(self):
        for path in [
                'some string',
                '/tmp/not-a-file',
                'http://not-a-url',
        ]:
            yield self.check_none, path

    def check_none(self, path):
        assert_raises(
            errors.LoadImageError,
            image_utils.load_image,
            path,
        )

    def test_good_file(self):
        for args in [
                # created mode, file extension, pixel value, loaded mode (expected)
                # Grayscale
                ('1',   'png',  1,              'L'),
                ('1',   'ppm',  1,              'L'),
                ('L',   'png',  127,            'L'),
                ('L',   'jpg',  127,            'L'),
                ('L',   'ppm',  127,            'L'),
                ('LA',  'png',  (127, 255),     'L'),
                # Color
                ('RGB', 'png',  (127, 127, 127),        'RGB'),
                ('RGB', 'jpg',  (127, 127, 127),        'RGB'),
                ('RGB', 'ppm',  (127, 127, 127),        'RGB'),
                ('RGBA', 'png',  (127, 127, 127, 255),   'RGB'),
                ('P',   'png',  127,                    'RGB'),
                ('CMYK', 'jpg',  (127, 127, 127, 127),   'RGB'),
                ('YCbCr', 'jpg', (127, 127, 127),        'RGB'),
        ]:
            yield self.check_good_file, args

    def check_good_file(self, args):
        orig_mode, suffix, pixel, new_mode = args

        orig = PIL.Image.new(orig_mode, (10, 10), pixel)

        # temp files cause permission errors so just generate the name
        tmp = tempfile.mkstemp(suffix='.' + suffix)
        orig.save(tmp[1])
        new = image_utils.load_image(tmp[1])
        try:
            # sometimes on windows the file is not closed yet
            # which can cause an exception
            os.close(tmp[0])
            os.remove(tmp[1])
        except:
            pass

        assert new is not None, 'load_image should never return None'
        assert new.mode == new_mode, 'Image mode should be "%s", not "%s\nargs - %s' % (new_mode, new.mode, args)

    @mock.patch('digits.utils.image.requests')
    def test_good_url(self, mock_requests):
        # requests
        response = mock.Mock()
        response.status_code = mock_requests.codes.ok
        img_file = os.path.join(
            os.path.dirname(digits.__file__),
            'static',
            'images',
            'mona_lisa.jpg',
        )
        with open(img_file, 'rb') as infile:
            response.content = infile.read()
        mock_requests.get.return_value = response

        img = image_utils.load_image('http://some-url')
        assert img is not None

    def test_corrupted_file(self):
        image = PIL.Image.fromarray(np.zeros((10, 10, 3), dtype=np.uint8))

        # Save image to a JPEG buffer.
        buffer_io = StringIO()
        image.save(buffer_io, format='jpeg')
        encoded = buffer_io.getvalue()
        buffer_io.close()

        # Corrupt the second half of the image buffer.
        size = len(encoded)
        corrupted = encoded[:size / 2] + encoded[size / 2:][::-1]

        # Save the corrupted image to a temporary file.
        fname = tempfile.mkstemp(suffix='.bin')
        f = os.fdopen(fname[0], 'wb')
        fname = fname[1]

        f.write(corrupted)
        f.close()

        assert_raises(
            errors.LoadImageError,
            image_utils.load_image,
            fname,
        )

        os.remove(fname)


class TestResizeImage():

    @classmethod
    def setup_class(cls):
        cls.np_gray = np.random.randint(0, 255, (10, 10)).astype('uint8')
        cls.pil_gray = PIL.Image.fromarray(cls.np_gray)
        cls.np_color = np.random.randint(0, 255, (10, 10, 3)).astype('uint8')
        cls.pil_color = PIL.Image.fromarray(cls.np_color)

    def test_configs(self):
        # lots of configs tested here
        for h in [10, 15]:
            for w in [10, 16]:
                for t in ['gray', 'color']:
                    # test channels=None (should autodetect channels)
                    if t == 'color':
                        s = (h, w, 3)
                    else:
                        s = (h, w)
                    yield self.verify_pil, (h, w, None, None, t, s)
                    yield self.verify_np, (h, w, None, None, t, s)

                    # test channels={3,1}
                    for c in [3, 1]:
                        for m in ['squash', 'crop', 'fill', 'half_crop']:
                            if c == 3:
                                s = (h, w, 3)
                            else:
                                s = (h, w)
                            yield self.verify_pil, (h, w, c, m, t, s)
                            yield self.verify_np, (h, w, c, m, t, s)

    def verify_pil(self, args):
        # pass a PIL.Image to resize_image and check the returned dimensions
        h, w, c, m, t, s = args
        if t == 'gray':
            i = self.pil_gray
        else:
            i = self.pil_color
        r = image_utils.resize_image(i, h, w, c, m)
        assert r.shape == s, 'Resized PIL.Image (orig=%s) should have been %s, but was %s %s' % (
            i.size, s, r.shape, self.args_to_str(args))
        assert r.dtype == np.uint8, 'image.dtype should be uint8, not %s' % r.dtype

    def verify_np(self, args):
        # pass a numpy.ndarray to resize_image and check the returned dimensions
        h, w, c, m, t, s = args
        if t == 'gray':
            i = self.np_gray
        else:
            i = self.np_color
        r = image_utils.resize_image(i, h, w, c, m)
        assert r.shape == s, 'Resized np.ndarray (orig=%s) should have been %s, but was %s %s' % (
            i.shape, s, r.shape, self.args_to_str(args))
        assert r.dtype == np.uint8, 'image.dtype should be uint8, not %s' % r.dtype

    def args_to_str(self, args):
        return """
        height=%s
        width=%s
        channels=%s
        resize_mode=%s
        image_type=%s
        shape=%s""" % args
