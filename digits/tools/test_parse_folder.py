# Copyright (c) 2014-2017, NVIDIA CORPORATION.  All rights reserved.

import itertools
import os
import shutil
import tempfile

import mock
from nose.tools import raises, assert_raises
import numpy as np
import PIL.Image

from . import parse_folder
from digits import test_utils


test_utils.skipIfNotFramework('none')


class TestUnescape():

    def test_hello(self):
        assert parse_folder.unescape('hello') == 'hello'

    def test_space(self):
        assert parse_folder.unescape('%20') == ' '


class TestValidateFolder():

    @classmethod
    def setUpClass(cls):
        cls.tmpdir = tempfile.mkdtemp()
        _handle, cls.tmpfile = tempfile.mkstemp(dir=cls.tmpdir)

    @classmethod
    def tearDownClass(cls):
        try:
            shutil.rmtree(cls.tmpdir)
        except:
            pass

    def test_dir(self):
        assert parse_folder.validate_folder(self.tmpdir) is True

    def test_file(self):
        assert parse_folder.validate_folder(self.tmpfile) is False

    def test_nonexistent_dir(self):
        assert parse_folder.validate_folder(os.path.abspath('not-a-directory')) is False

    def test_nonexistent_url(self):
        assert parse_folder.validate_folder('http://localhost/not-a-url') is False


class TestValidateOutputFile():

    @classmethod
    def setUpClass(cls):
        cls.tmpdir = tempfile.mkdtemp()
        _handle, cls.tmpfile = tempfile.mkstemp(dir=cls.tmpdir)

    @classmethod
    def tearDownClass(cls):
        try:
            shutil.rmtree(cls.tmpdir)
        except:
            pass

    def test_missing_file(self):
        assert parse_folder.validate_output_file(None) is True, 'all new files should be valid'

    def test_file(self):
        assert parse_folder.validate_output_file(os.path.join(self.tmpdir, 'output.txt')) is True

    @mock.patch('os.access')
    def test_local_file(self, mock_access):
        mock_access.return_value = True
        assert parse_folder.validate_output_file('not-a-file.txt') is True, 'relative paths should be accepted'

    @mock.patch('os.access')
    def test_not_writeable(self, mock_access):
        mock_access.return_value = False
        assert parse_folder.validate_output_file(self.tmpfile) is False, 'should not succeed without write permission'

    def test_existing_file(self):
        assert parse_folder.validate_output_file(self.tmpfile) is False

    def test_nonexistent_dir(self):
        assert parse_folder.validate_output_file(
            os.path.join(
                os.path.abspath('not-a-dir'),
                'output.txt'
            )
        ) is False


class TestValidateInputFile():

    @classmethod
    def setUpClass(cls):
        _handle, cls.tmpfile = tempfile.mkstemp()
        os.close(_handle)

    @classmethod
    def tearDownClass(cls):
        os.remove(cls.tmpfile)

    def test_missing_file(self):
        assert parse_folder.validate_input_file('not-a-file.txt') is False, 'should not pass on missing file'

    @mock.patch('os.access')
    def test_not_readable(self, mock_access):
        mock_access.return_value = False
        assert parse_folder.validate_input_file(self.tmpfile) is False, 'should not succeed without read permission'


class TestValidateRange():

    def test_no_range(self):
        assert parse_folder.validate_range(0) is True

    def test_min_less(self):
        assert parse_folder.validate_range(-1, min_value=0) is False

    def test_min_equal(self):
        assert parse_folder.validate_range(0, min_value=0) is True

    def test_min_more(self):
        assert parse_folder.validate_range(1, min_value=0) is True

    def test_max_less(self):
        assert parse_folder.validate_range(9, max_value=10) is True

    def test_max_equal(self):
        assert parse_folder.validate_range(10, max_value=10) is True

    def test_max_more(self):
        assert parse_folder.validate_range(11, max_value=10) is False

    def test_allow_none_true(self):
        assert parse_folder.validate_range(None, allow_none=True) is True

    def test_allow_none_false(self):
        assert parse_folder.validate_range(None, allow_none=False) is False

    def test_string(self):
        assert parse_folder.validate_range('foo') is False


@mock.patch('digits.tools.parse_folder.validate_output_file')
@mock.patch('digits.tools.parse_folder.validate_input_file')
class TestCalculatePercentages():

    @raises(AssertionError)
    def test_making_0(self, mock_input, mock_output):
        parse_folder.calculate_percentages(None, None, None, None, None, None, None)

    def test_making_1(self, mock_input, mock_output):
        mock_input.return_value = True
        mock_output.return_value = True

        expected_outputs = [
            ('train_file', (100, 0, 0)),
            ('val_file', (0, 100, 0)),
            ('test_file', (0, 0, 100))
        ]

        for supplied, expected in expected_outputs:
            args = {k: None for k in ['labels_file', 'train_file', 'percent_train',
                                      'val_file', 'percent_val', 'test_file', 'percent_test']}
            args.update({supplied: ''})

            output = parse_folder.calculate_percentages(**args)
            assert output == expected, 'expected output of {}, got {}'.format(output, expected)

    def test_making_2(self, mock_input, mock_output):
        mock_input.return_value = True
        mock_output.return_value = True

        permutes = itertools.combinations(['train', 'val', 'test'], 2)
        expected_outputs = itertools.izip(permutes, itertools.repeat((32, 68)))

        for supplied, expected in expected_outputs:
            args = {k: None for k in ['labels_file', 'train_file', 'percent_train',
                                      'val_file', 'percent_val', 'test_file', 'percent_test']}
            args.update({k + '_file': '' for k in supplied})
            args.update({'percent_' + k: v for k, v in itertools.izip(supplied, expected)})

            # Tricky line. itertools returns combinations in sorted order, always.
            # The order of the returned non-zero values should always be correct.
            output = [x for x in parse_folder.calculate_percentages(**args) if x != 0]
            assert output == list(expected), 'expected output of {}, got {}'.format(output, expected)

    def test_making_3_all_given(self, mock_input, mock_output):
        mock_input.return_value = True
        mock_output.return_value = True

        expected = (25, 30, 45)
        assert parse_folder.calculate_percentages(
            labels_file='not-a-file.txt',
            train_file='not-a-file.txt', percent_train=25,
            val_file='not-a-file.txt', percent_val=30,
            test_file='not-a-file.txt', percent_test=45
        ) == expected, 'Calculate percentages should return identical values of {}'.format(expected)

    def test_making_3_2_given(self, mock_input, mock_output):
        mock_input.return_value = True
        mock_output.return_value = True

        expected = 45
        assert parse_folder.calculate_percentages(
            labels_file='not-a-file.txt',
            train_file='not-a-file.txt', percent_train=25,
            val_file='not-a-file.txt', percent_val=30,
            test_file='not-a-file.txt', percent_test=None
        )[2] == expected, 'Calculate percentages should calculate third value of {}'.format(expected)

    @raises(AssertionError)
    def test_making_out_of_range(self, mock_input, mock_output):
        mock_input.return_value = True
        mock_output.return_value = True

        # should raise AssertionError because percentages not between 0-100 are invalid
        parse_folder.calculate_percentages(
            labels_file='not-a-file.txt',
            train_file='not-a-file.txt', percent_train=-1,
            val_file=None, percent_val=None,
            test_file=None, percent_test=None
        )


class TestParseWebListing():

    def test_non_url(self):
        for url in ['not-a-url', 'http://not-a-url', 'https://not-a-url']:
            yield self.check_url_raises, url

    def check_url_raises(self, url):
        assert_raises(Exception, parse_folder.parse_web_listing, url)

    def test_mock_url(self):
        for content, dirs, files in [
                # Nothing
                ('', [], []),
                # Apache 2.2.22
                (
                    '<head></head><body><table>\n \
<tr><td><a href="/home/">Parent</a></td></tr>\n \
<tr><td><a href="cat1/">cat1/</a></td><td>01-Jan-2015 12:34</td><td> - </td></tr>\n \
<tr><td><a href="cat2/">cat2/</a></td><td>02-Feb-2015 23:45</td><td> - </td></tr>\n \
<tr><td><a href="cat.jpg">cat.jpg</a></td><td>03-Mar-2015 1:23</td><td> 1 </td></tr>\n \
</table</body>\n',
                    ['cat1/', 'cat2/'],
                    ['cat.jpg'],
                ),
                # Apache 2.4.7
                (
                    '<html><head></head><body><table>\n \
<tr><td><a href="/home/">Parent</a></td></tr>\n \
<tr><td><a href="dog/">dog/</a></td><td>01-01-2015 12:34</td><td> - </td></tr>\n \
<tr><td><a href="dog1.jpeg">dog1.jpeg</a></td><td>02-02-2015 23:45</td><td> 1 </td></tr>\n \
<tr><td><a href="dog2.png">dog2.png</a></td><td>03-03-2015 1:23</td><td> 2 </td></tr>\n \
</table</body></html>\n',
                    ['dog/'],
                    ['dog1.jpeg', 'dog2.png'],
                ),
                # Nginx
                (
                    '<html><head></head><body>\n \
<a href="bird.jpg">bird.jpg</a> 01-Jan-1999 01:23 1\n \
<a href="birds/">birds/</a> 02-Feb-1999 12:34 -',
                    ['birds/'],
                    ['bird.jpg'],
                ),
        ]:
            with mock.patch('digits.tools.parse_folder.requests') as mock_requests:
                response = mock.Mock()
                response.status_code = mock_requests.codes.ok
                response.content = content
                mock_requests.get.return_value = response
                yield self.check_listing, (dirs, files)

    def check_listing(self, rc):
        assert parse_folder.parse_web_listing('any_url') == rc


class TestSplitIndices():

    def test_indices(self):
        for size in [5, 22, 32]:
            for percent_b in range(0, 100, 31):
                for percent_c in range(0, 100 - percent_b, 41):
                    yield self.check_split, size, percent_b, percent_c

    def check_split(self, size, pct_b, pct_c):
        ideala = size * float(100 - pct_b - pct_c) / 100.0
        idealb = size * float(100 - pct_c) / 100.0
        idxa, idxb = parse_folder.three_way_split_indices(size, pct_b, pct_c)

        assert abs(ideala - idxa) <= 2, 'split should be close to {}, is {}'.format(ideala, idxa)
        assert abs(idealb - idxb) <= 2, 'split should be close to {}, is {}'.format(idealb, idxb)


class TestParseFolder():

    def test_all_train(self):
        tmpdir = tempfile.mkdtemp()
        img = PIL.Image.fromarray(np.zeros((10, 10, 3), dtype='uint8'))
        classes = ['A', 'B', 'C']
        for cls in classes:
            os.makedirs(os.path.join(tmpdir, cls))
            img.save(os.path.join(tmpdir, cls, 'image1.png'))
            img.save(os.path.join(tmpdir, cls, 'image2.jpg'))

        labels_file = os.path.join(tmpdir, 'labels.txt')
        train_file = os.path.join(tmpdir, 'train.txt')

        parse_folder.parse_folder(tmpdir, labels_file, train_file=train_file,
                                  percent_train=100, percent_val=0, percent_test=0)

        with open(labels_file) as infile:
            parsed_classes = [line.strip() for line in infile]
            assert parsed_classes == classes, '%s != %s' % (parsed_classes, classes)

        shutil.rmtree(tmpdir)
