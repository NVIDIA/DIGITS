# Copyright (c) 2015-2017, NVIDIA CORPORATION.  All rights reserved.
from __future__ import absolute_import

import os
import re
import tempfile

import flask
import werkzeug.exceptions

from .forms import GenericImageModelForm
from .job import GenericImageModelJob
from digits.pretrained_model.job import PretrainedModelJob
from digits import extensions, frameworks, utils
from digits.config import config_value
from digits.dataset import GenericDatasetJob, GenericImageDatasetJob
from digits.inference import ImageInferenceJob
from digits.status import Status
from digits.utils import filesystem as fs
from digits.utils import constants
from digits.utils.forms import fill_form_if_cloned, save_form_to_job
from digits.utils.routing import request_wants_json, job_from_request
from digits.webapp import scheduler

blueprint = flask.Blueprint(__name__, __name__)


@blueprint.route('/new', methods=['GET'])
@blueprint.route('/new/<extension_id>', methods=['GET'])
@utils.auth.requires_login
def new(extension_id=None):
    """
    Return a form for a new GenericImageModelJob
    """
    form = GenericImageModelForm()
    form.dataset.choices = get_datasets(extension_id)
    form.standard_networks.choices = []
    form.previous_networks.choices = get_previous_networks()
    form.pretrained_networks.choices = get_pretrained_networks()
    prev_network_snapshots = get_previous_network_snapshots()

    # Is there a request to clone a job with ?clone=<job_id>
    fill_form_if_cloned(form)

    return flask.render_template(
        'models/images/generic/new.html',
        extension_id=extension_id,
        extension_title=extensions.data.get_extension(extension_id).get_title() if extension_id else None,
        form=form,
        frameworks=frameworks.get_frameworks(),
        previous_network_snapshots=prev_network_snapshots,
        previous_networks_fullinfo=get_previous_networks_fulldetails(),
        pretrained_networks_fullinfo=get_pretrained_networks_fulldetails(),
        multi_gpu=config_value('caffe')['multi_gpu'],
    )


@blueprint.route('<extension_id>.json', methods=['POST'])
@blueprint.route('<extension_id>', methods=['POST'], strict_slashes=False)
@blueprint.route('.json', methods=['POST'])
@blueprint.route('', methods=['POST'], strict_slashes=False)
@utils.auth.requires_login(redirect=False)
def create(extension_id=None):
    """
    Create a new GenericImageModelJob

    Returns JSON when requested: {job_id,name,status} or {errors:[]}
    """
    form = GenericImageModelForm()
    form.dataset.choices = get_datasets(extension_id)
    form.standard_networks.choices = []
    form.previous_networks.choices = get_previous_networks()
    form.pretrained_networks.choices = get_pretrained_networks()

    prev_network_snapshots = get_previous_network_snapshots()

    # Is there a request to clone a job with ?clone=<job_id>
    fill_form_if_cloned(form)

    if not form.validate_on_submit():
        if request_wants_json():
            return flask.jsonify({'errors': form.errors}), 400
        else:
            return flask.render_template(
                'models/images/generic/new.html',
                extension_id=extension_id,
                extension_title=extensions.data.get_extension(extension_id).get_title() if extension_id else None,
                form=form,
                frameworks=frameworks.get_frameworks(),
                previous_network_snapshots=prev_network_snapshots,
                previous_networks_fullinfo=get_previous_networks_fulldetails(),
                pretrained_networks_fullinfo=get_pretrained_networks_fulldetails(),
                multi_gpu=config_value('caffe')['multi_gpu'],
            ), 400

    datasetJob = scheduler.get_job(form.dataset.data)
    if not datasetJob:
        raise werkzeug.exceptions.BadRequest(
            'Unknown dataset job_id "%s"' % form.dataset.data)

    # sweeps will be a list of the the permutations of swept fields
    # Get swept learning_rate
    sweeps = [{'learning_rate': v} for v in form.learning_rate.data]
    add_learning_rate = len(form.learning_rate.data) > 1

    # Add swept batch_size
    sweeps = [dict(s.items() + [('batch_size', bs)]) for bs in form.batch_size.data for s in sweeps[:]]
    add_batch_size = len(form.batch_size.data) > 1
    n_jobs = len(sweeps)

    jobs = []
    for sweep in sweeps:
        # Populate the form with swept data to be used in saving and
        # launching jobs.
        form.learning_rate.data = sweep['learning_rate']
        form.batch_size.data = sweep['batch_size']

        # Augment Job Name
        extra = ''
        if add_learning_rate:
            extra += ' learning_rate:%s' % str(form.learning_rate.data[0])
        if add_batch_size:
            extra += ' batch_size:%d' % form.batch_size.data[0]

        job = None
        try:
            job = GenericImageModelJob(
                username=utils.auth.get_username(),
                name=form.model_name.data + extra,
                group=form.group_name.data,
                dataset_id=datasetJob.id(),
            )

            # get framework (hard-coded to caffe for now)
            fw = frameworks.get_framework_by_id(form.framework.data)

            pretrained_model = None
            # if form.method.data == 'standard':
            if form.method.data == 'previous':
                old_job = scheduler.get_job(form.previous_networks.data)
                if not old_job:
                    raise werkzeug.exceptions.BadRequest(
                        'Job not found: %s' % form.previous_networks.data)

                use_same_dataset = (old_job.dataset_id == job.dataset_id)
                network = fw.get_network_from_previous(old_job.train_task().network, use_same_dataset)

                for choice in form.previous_networks.choices:
                    if choice[0] == form.previous_networks.data:
                        epoch = float(flask.request.form['%s-snapshot' % form.previous_networks.data])
                        if epoch == 0:
                            pass
                        elif epoch == -1:
                            pretrained_model = old_job.train_task().pretrained_model
                        else:
                            for filename, e in old_job.train_task().snapshots:
                                if e == epoch:
                                    pretrained_model = filename
                                    break

                            if pretrained_model is None:
                                raise werkzeug.exceptions.BadRequest(
                                    "For the job %s, selected pretrained_model for epoch %d is invalid!"
                                    % (form.previous_networks.data, epoch))
                            if not (os.path.exists(pretrained_model)):
                                raise werkzeug.exceptions.BadRequest(
                                    "Pretrained_model for the selected epoch doesn't exist. "
                                    "May be deleted by another user/process. "
                                    "Please restart the server to load the correct pretrained_model details.")
                        break
            elif form.method.data == 'pretrained':
                pretrained_job = scheduler.get_job(form.pretrained_networks.data)
                model_def_path = pretrained_job.get_model_def_path()
                weights_path = pretrained_job.get_weights_path()

                network = fw.get_network_from_path(model_def_path)
                pretrained_model = weights_path

            elif form.method.data == 'custom':
                network = fw.get_network_from_desc(form.custom_network.data)
                pretrained_model = form.custom_network_snapshot.data.strip()
            else:
                raise werkzeug.exceptions.BadRequest(
                    'Unrecognized method: "%s"' % form.method.data)

            policy = {'policy': form.lr_policy.data}
            if form.lr_policy.data == 'fixed':
                pass
            elif form.lr_policy.data == 'step':
                policy['stepsize'] = form.lr_step_size.data
                policy['gamma'] = form.lr_step_gamma.data
            elif form.lr_policy.data == 'multistep':
                policy['stepvalue'] = form.lr_multistep_values.data
                policy['gamma'] = form.lr_multistep_gamma.data
            elif form.lr_policy.data == 'exp':
                policy['gamma'] = form.lr_exp_gamma.data
            elif form.lr_policy.data == 'inv':
                policy['gamma'] = form.lr_inv_gamma.data
                policy['power'] = form.lr_inv_power.data
            elif form.lr_policy.data == 'poly':
                policy['power'] = form.lr_poly_power.data
            elif form.lr_policy.data == 'sigmoid':
                policy['stepsize'] = form.lr_sigmoid_step.data
                policy['gamma'] = form.lr_sigmoid_gamma.data
            else:
                raise werkzeug.exceptions.BadRequest(
                    'Invalid learning rate policy')

            if config_value('caffe')['multi_gpu']:
                if form.select_gpu_count.data:
                    gpu_count = form.select_gpu_count.data
                    selected_gpus = None
                else:
                    selected_gpus = [str(gpu) for gpu in form.select_gpus.data]
                    gpu_count = None
            else:
                if form.select_gpu.data == 'next':
                    gpu_count = 1
                    selected_gpus = None
                else:
                    selected_gpus = [str(form.select_gpu.data)]
                    gpu_count = None

            # Set up data augmentation structure
            data_aug = {}
            data_aug['flip'] = form.aug_flip.data
            data_aug['quad_rot'] = form.aug_quad_rot.data
            data_aug['rot'] = form.aug_rot.data
            data_aug['scale'] = form.aug_scale.data
            data_aug['noise'] = form.aug_noise.data
            data_aug['hsv_use'] = form.aug_hsv_use.data
            data_aug['hsv_h'] = form.aug_hsv_h.data
            data_aug['hsv_s'] = form.aug_hsv_s.data
            data_aug['hsv_v'] = form.aug_hsv_v.data

            # Python Layer File may be on the server or copied from the client.
            fs.copy_python_layer_file(
                bool(form.python_layer_from_client.data),
                job.dir(),
                (flask.request.files[form.python_layer_client_file.name]
                 if form.python_layer_client_file.name in flask.request.files
                 else ''), form.python_layer_server_file.data)

            job.tasks.append(fw.create_train_task(
                job=job,
                dataset=datasetJob,
                train_epochs=form.train_epochs.data,
                snapshot_interval=form.snapshot_interval.data,
                learning_rate=form.learning_rate.data[0],
                lr_policy=policy,
                gpu_count=gpu_count,
                selected_gpus=selected_gpus,
                batch_size=form.batch_size.data[0],
                batch_accumulation=form.batch_accumulation.data,
                val_interval=form.val_interval.data,
                pretrained_model=pretrained_model,
                crop_size=form.crop_size.data,
                use_mean=form.use_mean.data,
                network=network,
                random_seed=form.random_seed.data,
                solver_type=form.solver_type.data,
                rms_decay=form.rms_decay.data,
                shuffle=form.shuffle.data,
                data_aug=data_aug,
            )
            )

            # Save form data with the job so we can easily clone it later.
            save_form_to_job(job, form)

            jobs.append(job)
            scheduler.add_job(job)
            if n_jobs == 1:
                if request_wants_json():
                    return flask.jsonify(job.json_dict())
                else:
                    return flask.redirect(flask.url_for('digits.model.views.show', job_id=job.id()))

        except:
            if job:
                scheduler.delete_job(job)
            raise

    if request_wants_json():
        return flask.jsonify(jobs=[j.json_dict() for j in jobs])

    # If there are multiple jobs launched, go to the home page.
    return flask.redirect('/')


def show(job, related_jobs=None):
    """
    Called from digits.model.views.models_show()
    """
    view_extensions = get_view_extensions()

    inference_form_html = None
    if isinstance(job.dataset, GenericDatasetJob):
        extension_class = extensions.data.get_extension(job.dataset.extension_id)
        if not extension_class:
            raise RuntimeError("Unable to find data extension with ID=%s"
                               % job.dataset.extension_id)
        extension_userdata = job.dataset.extension_userdata
        extension_userdata.update({'is_inference_db': True})
        extension = extension_class(**extension_userdata)

        form = extension.get_inference_form()
        if form:
            template, context = extension.get_inference_template(form)
            inference_form_html = flask.render_template_string(template, **context)

    return flask.render_template(
        'models/images/generic/show.html',
        job=job,
        view_extensions=view_extensions,
        related_jobs=related_jobs,
        inference_form_html=inference_form_html,
    )


@blueprint.route('/large_graph', methods=['GET'])
def large_graph():
    """
    Show the loss/accuracy graph, but bigger
    """
    job = job_from_request()

    return flask.render_template('models/images/generic/large_graph.html', job=job)


@blueprint.route('/infer_one.json', methods=['POST'])
@blueprint.route('/infer_one', methods=['POST', 'GET'])
def infer_one():
    """
    Infer one image
    """
    model_job = job_from_request()

    remove_image_path = False
    if 'image_path' in flask.request.form and flask.request.form['image_path']:
        image_path = flask.request.form['image_path']
    elif 'image_file' in flask.request.files and flask.request.files['image_file']:
        outfile = tempfile.mkstemp(suffix='.bin')
        flask.request.files['image_file'].save(outfile[1])
        image_path = outfile[1]
        os.close(outfile[0])
        remove_image_path = True
    else:
        raise werkzeug.exceptions.BadRequest('must provide image_path or image_file')

    epoch = None
    if 'snapshot_epoch' in flask.request.form:
        epoch = float(flask.request.form['snapshot_epoch'])

    layers = 'none'
    if 'show_visualizations' in flask.request.form and flask.request.form['show_visualizations']:
        layers = 'all'

    if 'dont_resize' in flask.request.form and flask.request.form['dont_resize']:
        resize = False
    else:
        resize = True

    # create inference job
    inference_job = ImageInferenceJob(
        username=utils.auth.get_username(),
        name="Infer One Image",
        model=model_job,
        images=[image_path],
        epoch=epoch,
        layers=layers,
        resize=resize,
    )

    # schedule tasks
    scheduler.add_job(inference_job)

    # wait for job to complete
    inference_job.wait_completion()

    # retrieve inference data
    inputs, outputs, model_visualization = inference_job.get_data()

    # set return status code
    status_code = 500 if inference_job.status == 'E' else 200

    # delete job folder and remove from scheduler list
    scheduler.delete_job(inference_job)

    if remove_image_path:
        os.remove(image_path)

    if inputs is not None and len(inputs['data']) == 1:
        image = utils.image.embed_image_html(inputs['data'][0])
        visualizations, header_html, app_begin_html, app_end_html = get_inference_visualizations(
            model_job.dataset,
            inputs,
            outputs)
        inference_view_html = visualizations[0]
    else:
        image = None
        inference_view_html = None
        header_html = None
        app_begin_html = None
        app_end_html = None

    if request_wants_json():
        return flask.jsonify({'outputs': dict((name, blob.tolist())
                                              for name, blob in outputs.iteritems())}), status_code
    else:
        return flask.render_template(
            'models/images/generic/infer_one.html',
            model_job=model_job,
            job=inference_job,
            image_src=image,
            inference_view_html=inference_view_html,
            header_html=header_html,
            app_begin_html=app_begin_html,
            app_end_html=app_end_html,
            visualizations=model_visualization,
            total_parameters=sum(v['param_count'] for v in model_visualization
                                 if v['vis_type'] == 'Weights'),
        ), status_code


@blueprint.route('/infer_extension.json', methods=['POST'])
@blueprint.route('/infer_extension', methods=['POST', 'GET'])
def infer_extension():
    """
    Perform inference using the data from an extension inference form
    """
    model_job = job_from_request()

    inference_db_job = None
    try:
        # create an inference database
        inference_db_job = create_inference_db(model_job)
        db_path = inference_db_job.get_feature_db_path(constants.TEST_DB)

        # create database creation job
        epoch = None
        if 'snapshot_epoch' in flask.request.form:
            epoch = float(flask.request.form['snapshot_epoch'])

        layers = 'none'
        if 'show_visualizations' in flask.request.form and flask.request.form['show_visualizations']:
            layers = 'all'

        # create inference job
        inference_job = ImageInferenceJob(
            username=utils.auth.get_username(),
            name="Inference",
            model=model_job,
            images=db_path,
            epoch=epoch,
            layers=layers,
            resize=False,
        )

        # schedule tasks
        scheduler.add_job(inference_job)

        # wait for job to complete
        inference_job.wait_completion()

    finally:
        if inference_db_job:
            scheduler.delete_job(inference_db_job)

    # retrieve inference data
    inputs, outputs, model_visualization = inference_job.get_data()

    # set return status code
    status_code = 500 if inference_job.status == 'E' else 200

    # delete job folder and remove from scheduler list
    scheduler.delete_job(inference_job)

    if outputs is not None and len(outputs) < 1:
        # an error occurred
        outputs = None

    if inputs is not None:
        keys = [str(idx) for idx in inputs['ids']]
        inference_views_html, header_html, app_begin_html, app_end_html = get_inference_visualizations(
            model_job.dataset,
            inputs,
            outputs)
    else:
        inference_views_html = None
        header_html = None
        keys = None
        app_begin_html = None
        app_end_html = None

    if request_wants_json():
        result = {}
        for i, key in enumerate(keys):
            result[key] = dict((name, blob[i].tolist()) for name, blob in outputs.iteritems())
        return flask.jsonify({'outputs': result}), status_code
    else:
        return flask.render_template(
            'models/images/generic/infer_extension.html',
            model_job=model_job,
            job=inference_job,
            keys=keys,
            inference_views_html=inference_views_html,
            header_html=header_html,
            app_begin_html=app_begin_html,
            app_end_html=app_end_html,
            visualizations=model_visualization,
            total_parameters=sum(v['param_count'] for v in model_visualization
                                 if v['vis_type'] == 'Weights'),
        ), status_code


@blueprint.route('/infer_db.json', methods=['POST'])
@blueprint.route('/infer_db', methods=['POST', 'GET'])
def infer_db():
    """
    Infer a database
    """
    model_job = job_from_request()

    if 'db_path' not in flask.request.form or flask.request.form['db_path'] is None:
        raise werkzeug.exceptions.BadRequest('db_path is a required field')

    db_path = flask.request.form['db_path']

    if not os.path.exists(db_path):
        raise werkzeug.exceptions.BadRequest('DB "%s" does not exit' % db_path)

    epoch = None
    if 'snapshot_epoch' in flask.request.form:
        epoch = float(flask.request.form['snapshot_epoch'])

    if 'dont_resize' in flask.request.form and flask.request.form['dont_resize']:
        resize = False
    else:
        resize = True

    # create inference job
    inference_job = ImageInferenceJob(
        username=utils.auth.get_username(),
        name="Infer Many Images",
        model=model_job,
        images=db_path,
        epoch=epoch,
        layers='none',
        resize=resize,
    )

    # schedule tasks
    scheduler.add_job(inference_job)

    # wait for job to complete
    inference_job.wait_completion()

    # retrieve inference data
    inputs, outputs, _ = inference_job.get_data()

    # set return status code
    status_code = 500 if inference_job.status == 'E' else 200

    # delete job folder and remove from scheduler list
    scheduler.delete_job(inference_job)

    if outputs is not None and len(outputs) < 1:
        # an error occurred
        outputs = None

    if inputs is not None:
        keys = [str(idx) for idx in inputs['ids']]
        inference_views_html, header_html, app_begin_html, app_end_html = get_inference_visualizations(
            model_job.dataset,
            inputs,
            outputs)
    else:
        inference_views_html = None
        header_html = None
        keys = None
        app_begin_html = None
        app_end_html = None

    if request_wants_json():
        result = {}
        for i, key in enumerate(keys):
            result[key] = dict((name, blob[i].tolist()) for name, blob in outputs.iteritems())
        return flask.jsonify({'outputs': result}), status_code
    else:
        return flask.render_template(
            'models/images/generic/infer_db.html',
            model_job=model_job,
            job=inference_job,
            keys=keys,
            inference_views_html=inference_views_html,
            header_html=header_html,
            app_begin_html=app_begin_html,
            app_end_html=app_end_html,
        ), status_code


@blueprint.route('/infer_many.json', methods=['POST'])
@blueprint.route('/infer_many', methods=['POST', 'GET'])
def infer_many():
    """
    Infer many images
    """
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

    if 'dont_resize' in flask.request.form and flask.request.form['dont_resize']:
        resize = False
    else:
        resize = True

    paths = []

    for line in image_list.readlines():
        line = line.strip()
        if not line:
            continue

        path = None
        # might contain a numerical label at the end
        match = re.match(r'(.*\S)\s+\d+$', line)
        if match:
            path = match.group(1)
        else:
            path = line

        if not utils.is_url(path) and image_folder and not os.path.isabs(path):
            path = os.path.join(image_folder, path)
        paths.append(path)

        if num_test_images is not None and len(paths) >= num_test_images:
            break

    # create inference job
    inference_job = ImageInferenceJob(
        username=utils.auth.get_username(),
        name="Infer Many Images",
        model=model_job,
        images=paths,
        epoch=epoch,
        layers='none',
        resize=resize,
    )

    # schedule tasks
    scheduler.add_job(inference_job)

    # wait for job to complete
    inference_job.wait_completion()

    # retrieve inference data
    inputs, outputs, _ = inference_job.get_data()

    # set return status code
    status_code = 500 if inference_job.status == 'E' else 200

    # delete job folder and remove from scheduler list
    scheduler.delete_job(inference_job)

    if outputs is not None and len(outputs) < 1:
        # an error occurred
        outputs = None

    if inputs is not None:
        paths = [paths[idx] for idx in inputs['ids']]
        inference_views_html, header_html, app_begin_html, app_end_html = get_inference_visualizations(
            model_job.dataset,
            inputs,
            outputs)
    else:
        inference_views_html = None
        header_html = None
        app_begin_html = None
        app_end_html = None

    if request_wants_json():
        result = {}
        for i, path in enumerate(paths):
            result[path] = dict((name, blob[i].tolist()) for name, blob in outputs.iteritems())
        return flask.jsonify({'outputs': result}), status_code
    else:
        return flask.render_template(
            'models/images/generic/infer_many.html',
            model_job=model_job,
            job=inference_job,
            paths=paths,
            inference_views_html=inference_views_html,
            header_html=header_html,
            app_begin_html=app_begin_html,
            app_end_html=app_end_html,
        ), status_code


def create_inference_db(model_job):
    # create instance of extension class
    extension_class = extensions.data.get_extension(model_job.dataset.extension_id)
    extension_userdata = model_job.dataset.extension_userdata
    extension_userdata.update({'is_inference_db': True})
    extension = extension_class(**extension_userdata)

    extension_form = extension.get_inference_form()
    extension_form_valid = extension_form.validate_on_submit()

    if not extension_form_valid:
        errors = extension_form.errors.copy()
        raise werkzeug.exceptions.BadRequest(repr(errors))

    extension.userdata.update(extension_form.data)

    # create job
    job = GenericDatasetJob(
        username=utils.auth.get_username(),
        name='Inference dataset',
        group=None,
        backend='lmdb',
        feature_encoding='none',
        label_encoding='none',
        batch_size=1,
        num_threads=1,
        force_same_shape=0,
        extension_id=model_job.dataset.extension_id,
        extension_userdata=extension.get_user_data(),
    )

    # schedule tasks and wait for job to complete
    scheduler.add_job(job)
    job.wait_completion()

    # check for errors
    if job.status != Status.DONE:
        msg = ""
        for task in job.tasks:
            if task.exception:
                msg = msg + task.exception
            if task.traceback:
                msg = msg + task.exception
        raise RuntimeError(msg)

    return job


def get_datasets(extension_id):
    if extension_id:
        jobs = [j for j in scheduler.jobs.values()
                if isinstance(j, GenericDatasetJob)
                and j.extension_id == extension_id
                and (j.status.is_running() or j.status == Status.DONE)]
    else:
        jobs = [j for j in scheduler.jobs.values()
                if (isinstance(j, GenericImageDatasetJob) or isinstance(j, GenericDatasetJob))
                and (j.status.is_running() or j.status == Status.DONE)]
    return [(j.id(), j.name())
            for j in sorted(jobs, cmp=lambda x, y: cmp(y.id(), x.id()))]


def get_inference_visualizations(dataset, inputs, outputs):
    # get extension ID from form and retrieve extension class
    if 'view_extension_id' in flask.request.form:
        view_extension_id = flask.request.form['view_extension_id']
        extension_class = extensions.view.get_extension(view_extension_id)
        if extension_class is None:
            raise ValueError("Unknown extension '%s'" % view_extension_id)
    else:
        # no view extension specified, use default
        extension_class = extensions.view.get_default_extension()
    extension_form = extension_class.get_config_form()

    # validate form
    extension_form_valid = extension_form.validate_on_submit()
    if not extension_form_valid:
        raise ValueError("Extension form validation failed with %s" % repr(extension_form.errors))

    # create instance of extension class
    extension = extension_class(dataset, **extension_form.data)

    visualizations = []
    # process data
    n = len(inputs['ids'])
    for idx in xrange(n):
        input_id = inputs['ids'][idx]
        input_data = inputs['data'][idx]
        output_data = {key: outputs[key][idx] for key in outputs}
        data = extension.process_data(
            input_id,
            input_data,
            output_data)
        template, context = extension.get_view_template(data)
        visualizations.append(
            flask.render_template_string(template, **context))
    # get header
    template, context = extension.get_header_template()
    header = flask.render_template_string(template, **context) if template else None
    app_begin, app_end = extension.get_ng_templates()
    return visualizations, header, app_begin, app_end


def get_previous_networks():
    return [(j.id(), j.name()) for j in sorted(
        [j for j in scheduler.jobs.values() if isinstance(j, GenericImageModelJob)],
        cmp=lambda x, y: cmp(y.id(), x.id())
    )
    ]


def get_previous_networks_fulldetails():
    return [(j) for j in sorted(
        [j for j in scheduler.jobs.values() if isinstance(j, GenericImageModelJob)],
        cmp=lambda x, y: cmp(y.id(), x.id())
    )
    ]


def get_previous_network_snapshots():
    prev_network_snapshots = []
    for job_id, _ in get_previous_networks():
        job = scheduler.get_job(job_id)
        e = [(0, 'None')] + [(epoch, 'Epoch #%s' % epoch)
                             for _, epoch in reversed(job.train_task().snapshots)]
        if job.train_task().pretrained_model:
            e.insert(0, (-1, 'Previous pretrained model'))
        prev_network_snapshots.append(e)
    return prev_network_snapshots


def get_pretrained_networks():
    return [(j.id(), j.name()) for j in sorted(
        [j for j in scheduler.jobs.values() if isinstance(j, PretrainedModelJob)],
        cmp=lambda x, y: cmp(y.id(), x.id())
    )
    ]


def get_pretrained_networks_fulldetails():
    return [(j) for j in sorted(
        [j for j in scheduler.jobs.values() if isinstance(j, PretrainedModelJob)],
        cmp=lambda x, y: cmp(y.id(), x.id())
    )
    ]


def get_view_extensions():
    """
    return all enabled view extensions
    """
    view_extensions = {}
    all_extensions = extensions.view.get_extensions()
    for extension in all_extensions:
        view_extensions[extension.get_id()] = extension.get_title()
    return view_extensions
