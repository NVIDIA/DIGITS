# Copyright (c) 2014-2015, NVIDIA CORPORATION.  All rights reserved.

from flask import render_template, url_for, request, abort

from digits.webapp import app, scheduler
import images.views
import images as dataset_images

NAMESPACE = '/datasets/'

@app.route(NAMESPACE + '<job_id>', methods=['GET'])
def datasets_show(job_id):
    job = scheduler.get_job(job_id)

    if job is None:
        abort(404)

    if isinstance(job, dataset_images.ImageClassificationDatasetJob):
        return dataset_images.classification.views.show(job)
    else:
        abort(404)

@app.route(NAMESPACE + 'summary', methods=['GET'])
def dataset_summary():
    job_id = request.args.get('job_id', '')
    if not job_id:
        return 'No job_id in request!'

    job = scheduler.get_job(job_id)

    return render_template('datasets/summary.html', dataset=job)

