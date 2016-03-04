# Copyright (c) 2016, NVIDIA CORPORATION.  All rights reserved.
from __future__ import absolute_import

import os
import re

import flask
import numpy as np
import werkzeug.exceptions

import digits

from digits import utils
from digits.inference import ImageInferenceClassifyManyJob
from digits.utils.routing import request_wants_json, job_from_request
from digits.webapp import scheduler

blueprint = flask.Blueprint(__name__, __name__)

"""
Read an image list and return lists of paths and ground truth
"""
def read_image_list(image_list, image_folder, num_test_images):
    paths = []
    ground_truths = []

    for line in image_list.readlines():
        line = line.strip()
        if not line:
            continue

        path = None
        # might contain a numerical label at the end
        match = re.match(r'(.*\S)\s+(\d+)$', line)
        if match:
            path = match.group(1)
            ground_truth = int(match.group(2))
        else:
            path = line
            ground_truth = None

        if not utils.is_url(path) and image_folder and not os.path.isabs(path):
            path = os.path.join(image_folder, path)
        paths.append(path)
        ground_truths.append(ground_truth)

        if num_test_images is not None and len(paths) >= num_test_images:
            break
    return paths, ground_truths

"""
Compute classification statistics
"""
def get_classification_stats(scores, ground_truths, labels):
    n_labels = len(labels)
    # take top 5
    indices = (-scores).argsort()[:, :5]

    # remove invalid ground truth
    ground_truths = [x if x is not None and (0 <= x < n_labels) else None for x in ground_truths]

    # how many pieces of ground truth to we have?
    n_ground_truth = len([1 for x in ground_truths if x is not None])
    show_ground_truth = n_ground_truth > 0

    # compute classifications and statistics
    classifications = []
    n_top1_accurate = 0
    n_top5_accurate = 0
    confusion_matrix = np.zeros((n_labels,n_labels), dtype=np.dtype(int))
    for image_index, index_list in enumerate(indices):
        result = []
        if ground_truths[image_index] is not None:
            if ground_truths[image_index] == index_list[0]:
                n_top1_accurate += 1
            if ground_truths[image_index] in index_list:
                n_top5_accurate += 1
            if (0 <= ground_truths[image_index] < n_labels) and (0 <= index_list[0] < n_labels):
               confusion_matrix[ground_truths[image_index], index_list[0]] += 1
        for i in index_list:
            # `i` is a category in labels and also an index into scores
            result.append((labels[i], round(100.0*scores[image_index, i],2)))
        classifications.append(result)

    # accuracy
    if show_ground_truth:
        top1_accuracy = round(100.0 * n_top1_accurate / n_ground_truth, 2)
        top5_accuracy = round(100.0 * n_top5_accurate / n_ground_truth, 2)
        per_class_accuracy = []
        for x in xrange(n_labels):
            n_examples = sum(confusion_matrix[x])
            per_class_accuracy.append(round(100.0 * confusion_matrix[x,x] / n_examples, 2) if n_examples > 0 else None)
    else:
        top1_accuracy = None
        top5_accuracy = None
        per_class_accuracy = None

    # replace ground truth indices with labels
    ground_truths = [labels[x] if x is not None and (0 <= x < n_labels ) else None for x in ground_truths]

    return (classifications,
            ground_truths,
            top1_accuracy,
            top5_accuracy,
            confusion_matrix,
            per_class_accuracy)

@blueprint.route('/classify_many.json', methods=['POST', 'GET'])
@blueprint.route('/classify_many', methods=['POST', 'GET'])
def classify_many():
    """
    Start a new classify_may job
    """

    # kicking off a new inference job
    model_job = job_from_request()
    image_list = flask.request.files.get('image_list')
    if not image_list:
        raise werkzeug.exceptions.BadRequest('image_list is a required field')

    if 'image_folder' in flask.request.form and flask.request.form['image_folder'].strip():
        image_folder = flask.request.form['image_folder']
        if not os.path.exists(image_folder):
            raise werkzeug.exceptions.BadRequest('image_folder "%s" does not exit' % image_folder)
    else:
        image_folder = None

    if 'num_test_images' in flask.request.form and flask.request.form['num_test_images'].strip():
        num_test_images = int(flask.request.form['num_test_images'])
    else:
        num_test_images = None

    epoch = None
    if 'snapshot_epoch' in flask.request.form:
        epoch = float(flask.request.form['snapshot_epoch'])

    paths, ground_truths = read_image_list(image_list, image_folder, num_test_images)

    # create inference job
    inference_job = ImageInferenceClassifyManyJob(
                username      = utils.auth.get_username(),
                name          = "Classify Many Images",
                model         = model_job,
                images        = paths,
                epoch         = epoch,
                layers        = 'none',
                ground_truths = ground_truths,
                )

    # schedule tasks
    scheduler.add_job(inference_job)

    if request_wants_json():
        return flask.jsonify(inference_job.json_dict())
    else:
        return flask.redirect(flask.url_for('digits.inference.views.show', job_id=inference_job.id()))

"""
Show image classification inference job
"""
def show(job):
    if isinstance(job, ImageInferenceClassifyManyJob):
        return show_classify_many(job)
    else:
        raise werkzeug.exceptions.BadRequest(
                    'Invalid job type')

"""
Show classify many inference job
"""
def show_classify_many(inference_job):

    # retrieve inference parameters
    model_job, paths, ground_truths = inference_job.get_parameters()

    if inference_job.status.is_running():
        # the inference job is still running
        if request_wants_json():
            return flask.jsonify(inference_job.json_dict())
        else:
            return flask.render_template('inference/images/classification/classify_many.html',
                model_job          = model_job,
                job                = inference_job,
                running            = True,
                )
    else:
        # the inference job has completed

        # retrieve inference data
        inputs, outputs, _ = inference_job.get_data()

        labels = model_job.train_task().get_labels()

        # delete job
        scheduler.delete_job(inference_job)

        if outputs is not None and len(outputs) >= 1:

            # convert to class probabilities for viewing
            last_output_name, last_output_data = outputs.items()[-1]
            if len(last_output_data) < 1:
                raise werkzeug.exceptions.BadRequest(
                        'Unable to classify any image from the file')

            if inputs is not None:
                # retrieve path and ground truth of images that were successfully processed
                paths = [paths[idx] for idx in inputs['ids']]
                ground_truths = [ground_truths[idx] for idx in inputs['ids']]

            # get statistics
            (classifications,
             ground_truths,
             top1_accuracy,
             top5_accuracy,
             confusion_matrix,
             per_class_accuracy) = get_classification_stats(last_output_data, ground_truths, labels)

        else:
            # an error occurred
            [classifications,
             ground_truths,
             top1_accuracy,
             top5_accuracy,
             confusion_matrix,
             per_class_accuracy] = 6 * [None]

        if request_wants_json():
            joined = dict(zip(paths, classifications))
            return flask.jsonify({'classifications': joined})
        else:
            return flask.render_template('inference/images/classification/classify_many.html',
                    model_job          = model_job,
                    job                = inference_job,
                    running            = False,
                    paths              = paths,
                    labels             = labels,
                    classifications    = classifications,
                    show_ground_truth  = top1_accuracy != None,
                    ground_truths      = ground_truths,
                    top1_accuracy      = top1_accuracy,
                    top5_accuracy      = top5_accuracy,
                    confusion_matrix   = confusion_matrix,
                    per_class_accuracy = per_class_accuracy,
                    )
