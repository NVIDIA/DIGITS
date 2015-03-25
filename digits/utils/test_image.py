# Copyright (c) 2014-2015, NVIDIA CORPORATION.  All rights reserved.

import tempfile
import StringIO

from nose.tools import assert_raises
import mock
import PIL.Image
import numpy as np
from skimage import data

from . import image as _

class TestLoadImage():

    def test_bad_path(self):
        """load_image with bad path"""
        for path in [
                'some string',
                '/tmp/not-a-file',
                'http://not-a-url',
                ]:
            yield self.check_none, path

    def check_none(self, path):
        assert _.load_image(path) is None

    @mock.patch('digits.utils.image.PIL.Image')
    @mock.patch('digits.utils.image.os.path')
    def test_good_file(self, mock_path, mock_Image):
        """load_image with good file"""
        mock_path.exists.return_value = True
        mock_Image.open = mock.Mock()
        assert _.load_image('/a/file') is not None
        mock_Image.open.assert_called_with('/a/file')

    @mock.patch('digits.utils.image.PIL.Image')
    @mock.patch('digits.utils.image.cStringIO')
    @mock.patch('digits.utils.image.requests')
    def test_good_url(self, mock_requests, mock_cStringIO, mock_Image):
        """load_image with good url"""
        # requests
        response = mock.Mock()
        response.status_code = mock_requests.codes.ok
        response.content = 'some content'
        mock_requests.get.return_value = response

        # cStringIO
        mock_cStringIO.StringIO = mock.Mock()
        mock_cStringIO.StringIO.return_value = 'an object'

        # Image
        mock_Image.open = mock.Mock()

        assert _.load_image('http://some-url') is not None
        mock_cStringIO.StringIO.assert_called_with('some content')
        mock_Image.open.assert_called_with('an object')

    def test_corrupted_file(self):
        """load_image with corrupted file"""
        lena = PIL.Image.fromarray(data.lena())

        # Save image to a JPEG buffer.
        buffer_io = StringIO.StringIO()
        lena.save(buffer_io, format='jpeg')
        encoded = buffer_io.getvalue()
        buffer_io.close()

        # Corrupt the second half of the image buffer.
        size = len(encoded)
        corrupted = encoded[:size/2] + encoded[size/2:][::-1]

        # Save the corrupted image to a temporary file.
        f = tempfile.NamedTemporaryFile(delete=False)
        f.write(corrupted)
        f.close()

        assert _.load_image(f.name) is None

class TestResizeImage():

    @classmethod
    def setup_class(cls):
        cls.image = PIL.Image.fromarray(data.lena())

    def test_configs(self):
        """Various resize_image configurations"""
        for h in [20, 30]:
            for w in [20]:
                for c in [3, 1, None]:
                    for m in [None, 'squash', 'crop', 'fill', 'half_crop']:
                        if c == 1:
                            s = (h, w)
                        else:
                            s = (h, w, 3)
                        yield self.verify_dims, (h, w, c, m, s)

    def verify_dims(self, args):
        h, w, c, m, s = args
        r = _.resize_image(self.image, h, w, c, m)
        assert r.shape == s, '%s != %s' % (r.shape, s)


