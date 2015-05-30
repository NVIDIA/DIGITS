# Copyright (c) 2014-2015, NVIDIA CORPORATION.  All rights reserved.

import os.path
import tempfile
import shutil

from nose.tools import assert_raises
import mock

from . import parse_folder as _

class TestUnescape():
    def test_hello(self):
        assert _.unescape('hello') == 'hello'

    def test_space(self):
        assert _.unescape('%20') == ' '

class TestValidateFolder():
    @classmethod
    def setUpClass(cls):
        cls.tmpdir = tempfile.mkdtemp()
        _handle, cls.tmpfile = tempfile.mkstemp(dir=cls.tmpdir)

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmpdir)

    def test_dir(self):
        assert _.validate_folder(self.tmpdir) == True

    def test_file(self):
        assert _.validate_folder(self.tmpfile) == False

    def test_nonexistent_dir(self):
        assert _.validate_folder(os.path.abspath('not-a-directory')) == False

    def test_nonexistent_url(self):
        assert _.validate_folder('http://localhost/not-a-url') == False

class TestValidateOutputFile():
    @classmethod
    def setUpClass(cls):
        cls.tmpdir = tempfile.mkdtemp()
        _handle, cls.tmpfile = tempfile.mkstemp(dir=cls.tmpdir)

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmpdir)

    def test_file(self):
        assert _.validate_output_file(os.path.join(self.tmpdir, 'output.txt')) == True

    def test_existing_file(self):
        assert _.validate_output_file(self.tmpfile) == False

    def test_nonexistent_dir(self):
        assert _.validate_output_file(
                os.path.join(
                    os.path.abspath('not-a-dir'),
                    'output.txt'
                    )
                ) == False

class TestValidateRange():
    def test_no_range(self):
        assert _.validate_range(0) == True

    def test_min_less(self):
        assert _.validate_range(-1, min_value=0) == False
    def test_min_equal(self):
        assert _.validate_range(0, min_value=0) == True
    def test_min_more(self):
        assert _.validate_range(1, min_value=0) == True

    def test_max_less(self):
        assert _.validate_range(9, max_value=10) == True
    def test_max_equal(self):
        assert _.validate_range(10, max_value=10) == True
    def test_max_more(self):
        assert _.validate_range(11, max_value=10) == False

    def test_allow_none_true(self):
        assert _.validate_range(None, allow_none=True) == True
    def test_allow_none_false(self):
        assert _.validate_range(None, allow_none=False) == False

    def test_string(self):
        assert _.validate_range('foo') == False


class TestParseWebListing():

    def test_non_url(self):
        """parse_web_listing with bad url"""
        for url in ['not-a-url', 'http://not-a-url', 'https://not-a-url']:
            yield self.check_url_raises, url

    def check_url_raises(self, url):
        assert_raises(Exception, _.parse_web_listing, url)

    def test_parse_web_listing(self):
        """parse_web_listing check output"""
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
            with mock.patch('tools.parse_folder.requests') as mock_requests:
                response = mock.Mock()
                response.status_code = mock_requests.codes.ok
                response.content = content
                mock_requests.get.return_value = response
                yield self.check_listing, (dirs, files)

    def check_listing(self, rc):
        assert _.parse_web_listing('any_url') == rc

