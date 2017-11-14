from s3_walker import S3Walker


class MockS3Walker(S3Walker):

    def __init__(self, classes=range(10)):  # endpoint, accesskey, secretkey):
        self.s3_dict = {}
        self.s3_dict['emptybucket'] = []

        data = []
        for prefix in ['train/', 'test/']:
            for i in classes:
                for j in range(1, 10):
                    data.append(prefix+str(i)+'/image'+str(j)+'.png')
        self.s3_dict['validbucket'] = data

        invalid_data = []
        for i in classes:
            for j in range(1, 10):
                invalid_data.append('train/'+str(i)+'/image'+str(j)+'.png')
        self.s3_dict['invalidbucket'] = invalid_data

    # not needed #
    def connect(self):
        pass

    # not needed #
    def head(self, bucket, key):
        pass

    # not needed #
    def get(self, bucket, key, filename):
        pass

    # not needed #
    def get_as_string(self, bucket, key):
        pass

    # not needed #
    def get_meta(self, bucket, key, meta):
        pass

    # not needed #
    def put(self, bucket, key, filename):
        pass

    # not needed #
    def listbucket(self, bucket, prefix='', max_size=1000, marker='', with_prefix=False):
        if len(self.s3_dict[bucket]) == 0:
            raise Exception('No keys in specified bucket')
        prefix_filtered_list = [k for k in self.s3_dict[bucket] if k.startswith(prefix)]

        if not with_prefix:
            return prefix_filtered_list

        result = []
        for full_name in prefix_filtered_list:  # train/1/image1.zip
            file_name = full_name[len(prefix):]  # 1/image1.zip
            label = file_name.split('/')[0]  # 1
            candidate_result = prefix + label  # train/1
            if candidate_result not in result:
                result.append(candidate_result)

        return result
