# Copyright (c) 2014-2017, NVIDIA CORPORATION.  All rights reserved.
from digits import test_utils
import os.path
import tempfile

try:
    from s3_walker import S3Walker
    from boto.s3.bucket import Bucket
    from boto.s3.key import Key
    import_failed = False
except ImportError:
    import_failed = True

global data, bucketData
data = 'content'
put_content = ''
if not import_failed:
    bucketData = [Key(name='key0'), Key(name='key1'), Key(name='key2')]


def mock_get_bucket(bucketname):
    bucket = Bucket(bucketname)
    bucket.get_key = mock_get_key
    bucket.list = mock_list_bucket
    return bucket


def mock_get_key(keyname):
    key = Key(name=keyname)
    key.set_contents_from_string('content')
    key.set_metadata('metadata_name', 'metadata_val')
    return key


def mock_set_contents_from_string(self, content):
    global data
    data = content


def mock_get_contents_as_string(self):
    return data


def mock_list_bucket(prefix='', delimiter='', marker=''):
    return bucketData


def mock_set_contents_from_filename(self, filename):
    file = open(filename, 'r')
    read = file.read()
    global put_content
    put_content = read
    file.close()


if not import_failed:
    Key.set_contents_from_string = mock_set_contents_from_string
    Key.get_contents_as_string = mock_get_contents_as_string
    Key.set_contents_from_filename = mock_set_contents_from_filename
    Bucket.list = mock_list_bucket


test_utils.skipIfNotFramework('none')


class TestInit():

    @classmethod
    def setUpClass(cls):
        if import_failed:
            test_utils.skipTest('Could not import s3_walker, most likely cause is Boto not installed')

    def test_valid_endpoint(self):
        walker = S3Walker('http://endpoint.com', 'accesskey', 'secretkey')
        assert walker.host == 'endpoint.com'
        assert walker.accesskey == 'accesskey'
        assert walker.secretkey == 'secretkey'

    def test_http_https_endpoint(self):

        # Test that HTTP endpoint is parsed properly and defaults to port 80
        http_walker = S3Walker('http://endpoint.com', 'accesskey', 'secretkey')
        assert http_walker.host == 'endpoint.com'
        assert http_walker.port == 80

        # Test that HTTPS endpoint is parsed properly and defaults to port 443
        https_walker = S3Walker('https://endpoint.com', 'accesskey', 'secretkey')
        assert https_walker.host == 'endpoint.com'
        assert https_walker.port == 443

    def test_port(self):
        # Validate port is parsed properly
        walker = S3Walker('http://endpoint.com:81', 'accesskey', 'secretkey')
        assert walker.port == 81

    def test_invalid_endpoint(self):
        # Validate exception is thrown for invalid endpoint (no http:// or https://)
        try:
            S3Walker('endpoint.com', 'accesskey', 'secretkey')
        except ValueError:
            return
        assert False


class TestGetMethods():

    @classmethod
    def setUpClass(cls):
        if import_failed:
            test_utils.skipTest('Could not import s3_walker, most likely cause is Boto not installed')
        cls.walker = S3Walker('http://endpoint.com', 'accesskey', 'secretkey')
        cls.walker.connect()
        cls.walker.conn.get_bucket = mock_get_bucket

    def test_head(self):
        # test head operation to confirm S3Walker requests correct key to S3 endpoint
        key = self.walker.head('bucket', 'key')
        assert key.name == 'key'

    def test_get_to_filename(self):
        # test get operation to confirm that key is properly stored to file
        filename = tempfile.mkstemp()
        self.walker.get('bucket', 'key', filename[1])
        assert os.path.isfile(filename[1])
        os.remove(filename[1])

    def test_get_as_string(self):
        # test get as string operation to confirm key is properly returned as string
        assert self.walker.get_as_string('bucket', 'key') == 'content'

    def test_get_meta(self):
        # test get metadata operation to confirm metadata is properly returned from key
        assert self.walker.get_meta('bucket', 'key', 'metadata_name') == 'metadata_val'

    def test_list_bucket(self):
        # test list bucket to confirm list of keys is returned from bucket
        keys = self.walker.listbucket('bucket')
        assert len(keys) == 3
        count = 0
        for key in keys:
            assert key == 'key' + str(count)
            count += 1


class TestPut():

    @classmethod
    def setUpClass(cls):
        if import_failed:
            test_utils.skipTest('Could not import s3_walker, most likely cause is Boto not installed')
        cls.walker = S3Walker('http://endpoint.com', 'accesskey', 'secretkey')
        cls.walker.connect()
        cls.walker.conn.get_bucket = mock_get_bucket

    def test_put(self):
        putData = tempfile.mkstemp()
        file = open(putData[1], 'w')
        expected_data = 'this the data for test put'
        file.write(expected_data)
        file.close()
        self.walker.put('bucket', 'key', putData[1])
        assert put_content == expected_data
        os.remove(putData[1])
