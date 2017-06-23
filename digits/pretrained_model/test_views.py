# Copyright (c) 2016-2017, NVIDIA CORPORATION.  All rights reserved.

import json
import os
import tempfile
import io
import tarfile

from bs4 import BeautifulSoup

import digits.webapp
import digits.dataset.images.classification.test_views
import digits.model.images.classification.test_views
from digits import test_utils
import digits.test_views


# May be too short on a slow system
TIMEOUT_DATASET = 45
TIMEOUT_MODEL = 60


class BaseTestUpload(digits.model.images.classification.test_views.BaseViewsTestWithModel):
    """
    Tests uploading Pretrained Models
    """

    def test_upload_manual(self):
        # job = digits.webapp.scheduler.get_job(self.model_id)
        job = digits.webapp.scheduler.get_job(self.model_id)

        if job is None:
            raise AssertionError('Failed To Create Job')

        # Write the stats of the job to json,
        # and store in tempfile (for archive)
        info = job.json_dict(verbose=False, epoch=-1)
        task = job.train_task()

        snapshot_filename = task.get_snapshot(-1)
        weights_file = open(snapshot_filename, 'r')
        model_def_file = open(os.path.join(job.dir(), task.model_file), 'r')
        labels_file = open(os.path.join(task.dataset.dir(), info["labels file"]), 'r')

        rv = self.app.post(
            '/pretrained_models/new',
            data={
                'weights_file': weights_file,
                'model_def_file': model_def_file,
                'labels_file': labels_file,
                'framework': info['framework'],
                'image_type': info["image dimensions"][2],
                'resize_mode': info["image resize mode"],
                'width': info["image dimensions"][0],
                'height': info["image dimensions"][1],
                'job_name': 'test_create_pretrained_model_job'
            }
        )
        s = BeautifulSoup(rv.data, 'html.parser')
        body = s.select('body')

        assert rv.status_code == 302, 'POST failed with %s\n\n%s' % (rv.status_code, body)

    def test_upload_archive(self):
        job = digits.webapp.scheduler.get_job(self.model_id)

        if job is None:
            raise AssertionError('Failed To Create Job')

        info = json.dumps(job.json_dict(verbose=False, epoch=-1), sort_keys=True, indent=4, separators=(',', ': '))
        info_io = io.BytesIO()
        info_io.write(info)

        tmp = tempfile.NamedTemporaryFile()

        tf = tarfile.open(fileobj=tmp, mode='w:')
        for path, name in job.download_files(-1):
            tf.add(path, arcname=name)

        tf_info = tarfile.TarInfo("info.json")
        tf_info.size = len(info_io.getvalue())
        info_io.seek(0)
        tf.addfile(tf_info, info_io)
        tmp.flush()
        tmp.seek(0)

        rv = self.app.post(
            '/pretrained_models/upload_archive',
            data={
                'archive': tmp
            }
        )
        s = BeautifulSoup(rv.data, 'html.parser')
        body = s.select('body')
        tmp.close()

        assert rv.status_code == 200, 'POST failed with %s\n\n%s' % (rv.status_code, body)


class TestCaffeUpload(BaseTestUpload, test_utils.CaffeMixin):
    pass


class TestTorchUpload(BaseTestUpload, test_utils.TorchMixin):
    pass
