# Copyright (c) 2014-2015, NVIDIA CORPORATION.  All rights reserved.

from flask import render_template, request, abort

from digits.webapp import app, scheduler, autodoc
import images.views
import images as dataset_images

NAMESPACE = '/datasets/'

@app.route(NAMESPACE + '<job_id>', methods=['GET'])
@autodoc('datasets')
def datasets_show(job_id):
    """
    Show a DatasetJob
    """
    job = scheduler.get_job(job_id)

    if job is None:
        abort(404)

    if isinstance(job, dataset_images.ImageClassificationDatasetJob):
        return dataset_images.classification.views.show(job)
    else:
        abort(404)

@app.route(NAMESPACE + 'summary', methods=['GET'])
@autodoc('datasets')
def dataset_summary():
    """
    Return a short HTML summary of a DatasetJob
    """
    job_id = request.args.get('job_id', '')
    if not job_id:
        return 'No job_id in request!'

    job = scheduler.get_job(job_id)

    return render_template('datasets/summary.html', dataset=job)

