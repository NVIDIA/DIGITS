# Copyright (c) 2016, NVIDIA CORPORATION.  All rights reserved.
from __future__ import absolute_import

import os
import re
import tempfile

import flask
import numpy as np
import werkzeug.exceptions

import digits

from digits import utils
from digits.inference import ImageInferenceClassifyOneJob, ImageInferenceClassifyManyJob, ImageInferenceTopNJob
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

@blueprint.route('/classify_one.json', methods=['POST'])
@blueprint.route('/classify_one', methods=['POST', 'GET'])
def classify_one():
    """
    Classify one image and return the top 5 classifications

    Returns JSON when requested: {predictions: {category: confidence,...}}
    """
    model_job = job_from_request()

    if 'image_url' in flask.request.form and flask.request.form['image_url']:
        image_path = flask.request.form['image_url']
    elif 'image_file' in flask.request.files and flask.request.files['image_file']:
        outfile = tempfile.mkstemp(suffix='.png')
        flask.request.files['image_file'].save(outfile[1])
        image_path = outfile[1]
        os.close(outfile[0])
    else:
        raise werkzeug.exceptions.BadRequest('must provide image_url or image_file')

    epoch = None
    if 'snapshot_epoch' in flask.request.form:
        epoch = float(flask.request.form['snapshot_epoch'])

    layers = 'none'
    if 'show_visualizations' in flask.request.form and flask.request.form['show_visualizations']:
        layers = 'all'

    # create inference job
    inference_job = ImageInferenceClassifyOneJob(
                username    = utils.auth.get_username(),
                name        = "Classify One Image",
                model       = model_job,
                images      = [image_path],
                epoch       = epoch,
                layers      = layers
                )

    # schedule tasks
    scheduler.add_job(inference_job)

    if request_wants_json():
        return flask.jsonify(inference_job.json_dict())
    else:
        return flask.redirect(flask.url_for('digits.inference.views.show', job_id=inference_job.id()))

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

@blueprint.route('/top_n', methods=['POST'])
def top_n():
    """
    Classify many images and show the top N images per category by confidence
    """
    model_job = job_from_request()

    image_list = flask.request.files['image_list']
    if not image_list:
        raise werkzeug.exceptions.BadRequest('File upload not found')

    epoch = None
    if 'snapshot_epoch' in flask.request.form:
        epoch = float(flask.request.form['snapshot_epoch'])
    if 'top_n' in flask.request.form and flask.request.form['top_n'].strip():
        top_n = int(flask.request.form['top_n'])
    else:
        top_n = 9

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

    paths, _ = read_image_list(image_list, image_folder, num_test_images)

    # create inference job
    inference_job = ImageInferenceTopNJob(
                username    = utils.auth.get_username(),
                name        = "TopN Image Classification",
                model       = model_job,
                images      = paths,
                epoch       = epoch,
                layers      = 'none',
                top_n       = top_n,
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
    if isinstance(job, ImageInferenceClassifyOneJob):
        return show_classify_one(job)
    elif isinstance(job, ImageInferenceClassifyManyJob):
        return show_classify_many(job)
    elif isinstance(job, ImageInferenceTopNJob):
        return show_top_n(job)
    else:
        raise werkzeug.exceptions.BadRequest(
                    'Invalid job type')

"""
Show classify one inference job
"""
def show_classify_one(inference_job):

    # retrieve inference parameters
    model_job, paths, _ = inference_job.get_parameters()

    if inference_job.status.is_running():
        # the inference job is still running
        if request_wants_json():
            return flask.jsonify(inference_job.json_dict())
        else:
            return flask.render_template('inference/images/classification/classify_one.html',
                model_job          = model_job,
                job                = inference_job,
                running            = True,
                )
    else:
        # the inference job has completed

        # retrieve inference data
        inputs, outputs, visualizations = inference_job.get_data()

        # delete job
        scheduler.delete_job(inference_job)

        # remove file (fails silently if a URL was provided)
        try:
            os.remove(paths[0])
        except:
            pass

        image = None
        predictions = []
        if inputs is not None and len(inputs['data']) == 1:
            image = utils.image.embed_image_html(inputs['data'][0])
            # convert to class probabilities for viewing
            last_output_name, last_output_data = outputs.items()[-1]

            if len(last_output_data) == 1:
                scores = last_output_data[0].flatten()
                indices = (-scores).argsort()
                labels = model_job.train_task().get_labels()
                predictions = []
                for i in indices:
                    predictions.append( (labels[i], scores[i]) )
                predictions = [(p[0], round(100.0*p[1],2)) for p in predictions[:5]]

        if request_wants_json():
            return flask.jsonify({'predictions': predictions})
        else:
            return flask.render_template('inference/images/classification/classify_one.html',
                    model_job       = model_job,
                    job             = inference_job,
                    image_src       = image,
                    predictions     = predictions,
                    visualizations  = visualizations,
                    total_parameters= sum(v['param_count'] for v in visualizations if v['vis_type'] == 'Weights'),
                    running         = False,
                    )

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
"""
show TopN classification job
"""
def show_top_n(inference_job):

    # retrieve inference parameters
    model_job, _, _, top_n = inference_job.get_parameters()

    if inference_job.status.is_running():
        # the inference job is still running
        if request_wants_json():
            return flask.jsonify(inference_job.json_dict())
        else:
            return flask.render_template('inference/images/classification/top_n.html',
                model_job          = model_job,
                job                = inference_job,
                running            = True,
                )
    else:

        # retrieve inference data
        inputs, outputs, _ = inference_job.get_data()

        # delete job
        scheduler.delete_job(inference_job)

        results = None
        if outputs is not None and len(outputs) > 0:
            # convert to class probabilities for viewing
            last_output_name, last_output_data = outputs.items()[-1]
            scores = last_output_data

            if scores is None:
                raise RuntimeError('An error occured while processing the images')

            labels = model_job.train_task().get_labels()
            images = inputs['data']
            indices = (-scores).argsort(axis=0)[:top_n]
            results = []
            # Can't have more images per category than the number of images
            images_per_category = min(top_n, len(images))
            for i in xrange(indices.shape[1]):
                result_images = []
                for j in xrange(images_per_category):
                    result_images.append(images[indices[j][i]])
                results.append((
                        labels[i],
                        utils.image.embed_image_html(
                            utils.image.vis_square(np.array(result_images),
                                colormap='white')
                            )
                        ))

        return flask.render_template('inference/images/classification/top_n.html',
                model_job       = model_job,
                job             = inference_job,
                results         = results,
                running         = False,
                )
