# Copyright (c) 2016, NVIDIA CORPORATION.  All rights reserved.
"""
Loads old DIGITS data (pickle files) into SQL
"""

import copy
from datetime import datetime
import json
import os.path

from google.protobuf import text_format

from digits import database
from digits.database.adapter import db
from digits.webapp import app, scheduler

def print_keys():
    jd = {}
    td = {}

    for job_id, job in scheduler.jobs.iteritems():
        jname = type(job).__name__
        if jname not in jd:
            jd[jname] = set()

        jd[jname].update(set(job.__dict__.keys()))

        for task in job.tasks:
            tname = type(task).__name__
            if tname not in td:
                td[tname] = set()

            td[tname].update(set(task.__dict__.keys()))

    # Calculate keys shared by all jobs
    jshared = None
    for t, s in jd.iteritems():
        if jshared is None:
            jshared = copy.deepcopy(s)
        else:
            jshared.intersection_update(s)

    print '--- Job'
    for v in sorted(jshared):
        print v
    print

    for t, s in jd.iteritems():
        print '---', t
        for v in sorted(s.difference(jshared)):
            print v
        print

    # Calculate keys shared by all tasks
    tshared = None
    for t, s in td.iteritems():
        if tshared is None:
            tshared = copy.deepcopy(s)
        else:
            tshared.intersection_update(s)

    print '--- Task'
    for v in sorted(tshared):
        print v
    print

    for t, s in td.iteritems():
        print '---', t
        for v in sorted(s.difference(tshared)):
            print v
        print

def delete_everything():
    with app.app_context():
        print 'Delete everything ...'
        for table in reversed(db.metadata.sorted_tables):
            db.session.execute(table.delete())
        db.session.commit()

def load():
    delete_everything()

    with app.app_context():
        for job_id in sorted(scheduler.jobs):
            print 'Loading %s ...' % job_id
            job = scheduler.jobs[job_id]
            process_job(job)
            db.session.commit()

        print 'Done.'

def _relative_path(obj, filename):
    """
    Utility for turning a Job or Task file into a path that's
    relative to the specific job directory
    """
    if os.path.isabs(filename):
        return obj.path(filename)
    else:
        return obj.path(filename, relative=True)
#    return os.path.join(
#        *os.path.split(obj.path(filename, relative=True))[1:])

def process_job(pkl_job):
    from digits.dataset.images.classification.job import ImageClassificationDatasetJob
    from digits.dataset.images.generic.job import GenericImageDatasetJob
    from digits.model.images.classification.job import ImageClassificationModelJob
    from digits.model.images.generic.job import GenericImageModelJob


    job = database.Job()
    job.directory = os.path.basename(pkl_job.dir()) + '/'
    job.name = pkl_job.name()
    job.notes = pkl_job.notes()
    db.session.add(job)

    if isinstance(pkl_job, ImageClassificationDatasetJob):
        job.type = 'create-dataset'
        job.set_attribute('dataset_type', 'classification')
    elif isinstance(pkl_job, GenericImageDatasetJob):
        job.type = 'create-dataset'
        job.set_attribute('dataset_type', 'generic')
    elif isinstance(pkl_job, ImageClassificationModelJob):
        job.type = 'train-model'
        job.set_attribute('network_type', 'classification')
    elif isinstance(pkl_job, GenericImageModelJob):
        job.type = 'train-model'
        job.set_attribute('network_type', 'generic')
    else:
        raise AssertionError('Unknown job type %s' % type(pkl_job).__name__)

    # User
    if pkl_job.username:
        user = database.User.query.filter_by(name=pkl_job.username).first()
        if not user:
            user = database.User()
            user.name = pkl_job.username
        job.user = user

    # JobStatusUpdate
    for status, timestamp in pkl_job.status_history:
        job.status_updates.append(
            database.JobStatusUpdate(
                status=status.val,
                timestamp=datetime.utcfromtimestamp(timestamp)
            ))

    # Task
    objid_to_sql = {} # Mapping is useful later for Task.parents
    for pkl_task in pkl_job.tasks:
        task = process_task(job, pkl_task)
        objid_to_sql[id(pkl_task)] = task

    # Task.parents
    for pkl_task in pkl_job.tasks:
        if pkl_task.parents:
            task = objid_to_sql[id(pkl_task)]
            for parent_pkl_task in pkl_task.parents:
                task.parents.append(objid_to_sql[id(parent_pkl_task)])

    if isinstance(pkl_job, GenericImageDatasetJob):
        db.session.commit()
        for task in job.tasks:
            if task.type == 'analyze-db':
                for dataset in task.datasets:
                    phase = dataset.get_attribute('intended_phase')
                    blob_name = dataset.get_attribute('intended_blob_name')
                    if phase == 'train' and blob_name == 'data':
                        dataset.set_file('mean:binaryproto',
                                         _relative_path(pkl_job, pkl_job.mean_file))

    elif isinstance(pkl_job, ImageClassificationModelJob):
        # Training.datasets
        dataset_job_dir = os.path.basename(pkl_job.dataset.dir()) + '/'
        dataset_job = database.Job.query.filter_by(directory=dataset_job_dir).first()
        assert dataset_job is not None
        assert len(job.tasks) == 1
        assert len(job.tasks[0].trainings) == 1
        for task in dataset_job.tasks:
            if task.type == 'create-db':
                assert len(task.datasets) == 1
                job.tasks[0].trainings[0].datasets.append(task.datasets[0])

    elif isinstance(pkl_job, GenericImageModelJob):
        # Training.datasets
        dataset_job_dir = os.path.basename(pkl_job.dataset.dir()) + '/'
        dataset_job = database.Job.query.filter_by(directory=dataset_job_dir).first()
        assert dataset_job is not None
        assert len(job.tasks) == 1
        assert len(job.tasks[0].trainings) == 1
        for task in dataset_job.tasks:
            if task.type == 'analyze-db':
                assert len(task.datasets) == 1
                job.tasks[0].trainings[0].datasets.append(task.datasets[0])

    if pkl_job.exception is not None:
        job.set_attribute('exception', pkl_job.exception)

    if hasattr(pkl_job, 'form_data'):
        filename = 'form_data.json'
        with open(pkl_job.path(filename), 'w') as outfile:
            json.dump(pkl_job.form_data, outfile, sort_keys=True,
                         indent=4, separators=(',',': '))
        job.set_file('form_data', _relative_path(pkl_job, filename))


def process_task(job, pkl_task):
    from digits.dataset import tasks as dataset_tasks
    from digits.model import tasks as model_tasks

    task = database.Task(progress=pkl_task.progress)
    task.job = job

    # TaskStatusUpdate
    for status, timestamp in pkl_task.status_history:
        task.status_updates.append(
            database.TaskStatusUpdate(
                status=status.val,
                timestamp=datetime.utcfromtimestamp(timestamp)
            ))

    if pkl_task.exception is not None:
        task.set_attribute('error_message', pkl_task.exception)

    if pkl_task.traceback is not None:
        filename = 'error_info.txt'
        with open(pkl_task.path(filename), 'w') as outfile:
            outfile.write(pkl_task.traceback + '\n')
        task.set_file('error_info', _relative_path(pkl_task, filename))

    if isinstance(pkl_task, dataset_tasks.AnalyzeDbTask):
        process_analyze_db_task(pkl_task, task)
    elif isinstance(pkl_task, dataset_tasks.CreateDbTask):
        process_create_db_task(pkl_task, task)
    elif isinstance(pkl_task, dataset_tasks.ParseFolderTask):
        process_parse_folder_task(pkl_task, task)
    elif isinstance(pkl_task, model_tasks.TrainTask):
        process_train_task(pkl_task, task)
    else:
        raise AssertionError('Unknown task type %s' % type(pkl_task).__name__)

    return task


def process_analyze_db_task(pkl_task, task):
    task.type = 'analyze-db'
    if hasattr(pkl_task, 'force_same_shape'):
        task.set_attribute('input:force_same_shape', str(bool(pkl_task.force_same_shape)))
    if hasattr(pkl_task, 'analyze_db_log_file') and pkl_task.analyze_db_log_file:
        task.set_file('log', _relative_path(pkl_task, pkl_task.analyze_db_log_file))

    dataset = database.Dataset()
    dataset.task = task

    dataset.set_file('database', _relative_path(pkl_task, pkl_task.database))
    if hasattr(pkl_task, 'backend') and pkl_task.backend:
        dataset.set_attribute('backend', pkl_task.backend)

    intended_phase = pkl_task.purpose.split()[0]
    if intended_phase == 'Training':
        intended_phase = 'train'
    elif intended_phase == 'Validation':
        intended_phase = 'val'
    else:
        assert False, 'intended_phase = ' % intended_phase
    dataset.set_attribute('intended_phase', intended_phase)
    intended_blob_name = pkl_task.purpose.split()[1]
    if intended_blob_name == 'Images':
        intended_blob_name = 'data'
    elif intended_blob_name == 'Labels':
        intended_blob_name = 'label'
    else:
        assert False, 'intended_blob_name = ' % intended_blob_name
    dataset.set_attribute('intended_blob_name', intended_blob_name)
    dataset.set_attribute('count', pkl_task.image_count)
    dims = '%d,%d,%d' % (
        pkl_task.image_channels,
        pkl_task.image_height,
        pkl_task.image_width,
    )
    dataset.set_attribute('dimensions:%s' % intended_blob_name, dims)

    return task


def process_create_db_task(pkl_task, task):
    task.type = 'create-db'
    if hasattr(pkl_task, 'create_db_log_file') and pkl_task.create_db_log_file:
        task.set_file('log', _relative_path(pkl_task, pkl_task.create_db_log_file))

    dataset = database.Dataset()
    dataset.task = task

    intended_phase = None
    if 'train' in pkl_task.db_name.lower():
        intended_phase = 'train'
    elif 'val' in pkl_task.db_name.lower():
        intended_phase = 'val'
    elif 'test' in pkl_task.db_name.lower():
        intended_phase = 'test'
    if intended_phase is not None:
        dataset.set_attribute('intended_phase', intended_phase)
    dataset.set_attribute('compression', pkl_task.compression)
    dataset.set_attribute('encoding', pkl_task.encoding)
    dataset.set_attribute('resize_mode', pkl_task.resize_mode)
    if hasattr(pkl_task, 'shuffle'):
        dataset.set_attribute('shuffle', str(pkl_task.shuffle))
    if hasattr(pkl_task, 'image_channel_order') and pkl_task.image_channel_order:
        dataset.set_attribute('image_channel_order', pkl_task.image_channel_order)
    dataset.set_attribute('intended_blob_name', 'data,label')
    dataset.set_attribute('count', pkl_task.entries_count)
    dims = '%d,%d,%d' % (
        pkl_task.image_dims[2], # channels
        pkl_task.image_dims[0], # width
        pkl_task.image_dims[1], # height
    )
    dataset.set_attribute('dimensions:data', dims)
    dataset.set_attribute('dimension:label', 1)
    if hasattr(pkl_task, 'backend') and pkl_task.backend:
        dataset.set_attribute('backend', pkl_task.backend)
    if hasattr(pkl_task, 'image_folder') and pkl_task.image_folder:
            dataset.set_attribute('input:filelist:rootfolder', pkl_task.image_folder)

    if hasattr(pkl_task, 'distribution') and pkl_task.distribution:
        # write to file rather than saving hundreds of attributes
        filename = '%s_distribution.txt' % intended_phase
        with open(pkl_task.path(filename), 'w') as outfile:
            d = pkl_task.distribution
            for k in sorted(d.keys()):
                outfile.write('%d\n' % d[k])
        dataset.set_file('distribution', _relative_path(pkl_task, filename))

    # files
    dataset.set_file('database', _relative_path(pkl_task, pkl_task.db_name))
    dataset.set_file('input:filelist', _relative_path(pkl_task, pkl_task.input_file))
    dataset.set_file('labels', _relative_path(pkl_task, pkl_task.labels_file))
    if hasattr(pkl_task, 'textfile') and pkl_task.textfile:
        assert pkl_task.backend == 'hdf5'
        dataset.set_file('hdf5_filelist', _relative_path(pkl_task, pkl_task.textfile))

    # TODO: start saving means for other phases
    if intended_phase == 'train':
        jpg_path = pkl_task.path('mean.jpg')
        assert os.path.exists(jpg_path), 'mean.jpg not found'
        binaryproto_path = pkl_task.path('mean.binaryproto')
        assert os.path.exists(binaryproto_path), 'mean.binaryproto not found'
        dataset.set_file('mean:jpg', _relative_path(pkl_task, 'mean.jpg'))
        dataset.set_file('mean:binaryproto', _relative_path(pkl_task, pkl_task.mean_file))

    return task


def process_parse_folder_task(pkl_task, task):
    task.type = 'parse-folder'
    task.set_attribute('input:folder', pkl_task.folder)
    # input
    if hasattr(pkl_task, 'max_per_category') and pkl_task.max_per_category is not None:
        task.set_attribute('input:per_category:max', pkl_task.max_per_category)
    if hasattr(pkl_task, 'min_per_category') and pkl_task.min_per_category is not None and pkl_task.min_per_category != 2:
        task.set_attribute('input:per_category:min', pkl_task.min_per_category)
    if hasattr(pkl_task, 'percent_val') and pkl_task.percent_val:
        task.set_attribute('input:percentage:val', pkl_task.percent_val)
    if hasattr(pkl_task, 'percent_test') and pkl_task.percent_test:
        task.set_attribute('input:percentage:test', pkl_task.percent_test)
    # output
    if hasattr(pkl_task, 'label_count') and pkl_task.label_count:
        task.set_attribute('output:count:label', pkl_task.label_count)
    if hasattr(pkl_task, 'train_count') and pkl_task.train_count:
        task.set_attribute('output:count:train', pkl_task.train_count)
    if hasattr(pkl_task, 'val_count') and pkl_task.val_count:
        task.set_attribute('output:count:val', pkl_task.val_count)
    if hasattr(pkl_task, 'test_count') and pkl_task.test_count:
        task.set_attribute('output:count:test', pkl_task.test_count)

    task.set_file('labels', pkl_task.labels_file)
    if hasattr(pkl_task, 'train_file'):
        task.set_file('train', pkl_task.train_file)
    if hasattr(pkl_task, 'val_file'):
        task.set_file('val', pkl_task.val_file)
    if hasattr(pkl_task, 'test_file'):
        task.set_file('test', pkl_task.test_file)

    return task


def process_train_task(pkl_task, task):
    from digits.model import tasks as model_tasks

    training = database.Training()
    training.task = task

    if isinstance(pkl_task, model_tasks.CaffeTrainTask):
        process_caffe_train_task(pkl_task, task, training)
    elif isinstance(pkl_task, model_tasks.TorchTrainTask):
        process_torch_train_task(pkl_task, task, training)

    training.set_attribute('learning_rate', pkl_task.learning_rate)

    for key, val in pkl_task.lr_policy.iteritems():
        training.set_attribute('lr_policy:%s' % key, val)

    if hasattr(pkl_task, 'batch_size') and pkl_task.batch_size is not None:
        training.set_attribute('input:batch_size', pkl_task.batch_size)
    if hasattr(pkl_task, 'crop_size') and pkl_task.crop_size is not None:
        training.set_attribute('input:crop_size', pkl_task.crop_size)

    gpu_count = None
    if hasattr(pkl_task, 'trained_on_cpu') and pkl_task.trained_on_cpu is True:
        gpu_count = 0
    elif hasattr(pkl_task, 'gpu_count') and pkl_task.gpu_count is not None:
        gpu_count = pkl_task.gpu_count
    elif hasattr(pkl_task, 'selected_gpus') and pkl_task.selected_gpus:
        gpu_count = len(pkl_task.selected_gpus)
    if gpu_count is not None:
        training.set_attribute('gpu_count', gpu_count)

    # training outputs
    if pkl_task.train_outputs:
        train_outputs = pkl_task.train_outputs.keys()
        train_outputs.remove('epoch')
        for name in train_outputs:
            training.set_attribute('output_type:%s' % name, pkl_task.train_outputs[name].kind)
            for i, epoch in enumerate(pkl_task.train_outputs['epoch'].data):
                try:
                    values = pkl_task.train_outputs[name].data[i]
                    if not isinstance(values, list):
                        values = [values]
                    for v in values:
                        training.updates.append(database.TrainingUpdate(
                            epoch=epoch,
                            phase='train',
                            name=name,
                            value=v,
                        ))
                except IndexError as e:
                    print type(e).__name__, e, i, len(pkl_task.train_outputs[name].data)
    if pkl_task.val_outputs:
        val_outputs = pkl_task.val_outputs.keys()
        val_outputs.remove('epoch')
        for name in val_outputs:
            training.set_attribute('output_type:%s' % name, pkl_task.val_outputs[name].kind)
            for i, epoch in enumerate(pkl_task.val_outputs['epoch'].data):
                try:
                    values = pkl_task.val_outputs[name].data[i]
                    if not isinstance(values, list):
                        values = [values]
                    for v in values:
                        training.updates.append(database.TrainingUpdate(
                            epoch=epoch,
                            phase='val',
                            name=name,
                            value=v,
                        ))
                except IndexError as e:
                    print type(e).__name__, e, i, len(pkl_task.val_outputs[name].data)

    # pretrained model[s]
    if pkl_task.pretrained_model:
        for filename in pkl_task.pretrained_model.split(':'):
            filename = filename.strip()
            training.set_file('pretrained_model', _relative_path(pkl_task, filename))

    if hasattr(pkl_task, 'random_seed') and pkl_task.random_seed is not None:
        training.set_attribute('random_seed', pkl_task.random_seed)
    if hasattr(pkl_task, 'solver_type'):
        training.set_attribute('solver_type', pkl_task.solver_type)
    if hasattr(pkl_task, 'use_mean'):
        training.set_attribute('mean_method', pkl_task.use_mean)

    training.set_attribute('train_epochs', pkl_task.train_epochs)
    training.set_attribute('val_interval', pkl_task.val_interval)
    training.set_attribute('snapshot_interval', pkl_task.snapshot_interval)
    if pkl_task.snapshot_prefix != 'snapshot':
        training.set_attribute('snapshot_prefix', pkl_task.snapshot_prefix)

    return task


def process_caffe_train_task(pkl_task, task, training):
    """
    Handle stuff specific to caffe
    """
    task.type= 'train-caffe'
    training.set_attribute('framework', 'caffe')
    training.set_file('network_architecture:training',
                      _relative_path(pkl_task, pkl_task.train_val_file))
    training.set_file('network_architecture:inference',
                      _relative_path(pkl_task, pkl_task.deploy_file))
    inference_architecture_file = pkl_task.deploy_file
    training.set_file('solver', _relative_path(pkl_task, pkl_task.solver_file))

    has_caffe_log_file = hasattr(pkl_task, 'caffe_log_file') and pkl_task.caffe_log_file
    has_log_file = hasattr(pkl_task, 'log_file') and pkl_task.log_file
    if has_caffe_log_file and has_log_file:
        assert (pkl_task.path(pkl_task.caffe_log_file)
                == pkl_task.path(pkl_task.log_file)), 'mismatched log files'
    if has_log_file:
        task.set_file('log', _relative_path(pkl_task, pkl_task.log_file))
    elif has_caffe_log_file:
        task.set_file('log', _relative_path(pkl_task, pkl_task.caffe_log_file))

    # original network
    filename = 'original.prototxt'
    with open(pkl_task.path(filename), 'w') as outfile:
        text_format.PrintMessage(pkl_task.network, outfile)
    training.set_file('network_architecture:input', _relative_path(pkl_task, filename))

    # snapshots
    for snapshot_path, snapshot_epoch in pkl_task.snapshots:
        model = database.Model()
        model.training = training

        model.set_attribute('epoch', snapshot_epoch)
        model.set_file('weights', _relative_path(pkl_task, snapshot_path))
        model.set_file('network_architecture', _relative_path(pkl_task, inference_architecture_file))

    return task


def process_torch_train_task(pkl_task, task, training):
    """
    Handle stuff specific to torch
    """
    task.type= 'train-torch'
    training.set_attribute('framework', 'torch')
    training.set_file('network_architecture', _relative_path(pkl_task, pkl_task.model_file))
    inference_architecture_file = pkl_task.model_file
    if hasattr(pkl_task, 'shuffle') and pkl_task.shuffle is not None:
        training.set_attribute('shuffle_data', str(pkl_task.shuffle))

    has_torch_log_file = hasattr(pkl_task, 'torch_log_file') and pkl_task.torch_log_file
    has_log_file = hasattr(pkl_task, 'log_file') and pkl_task.log_file
    if has_torch_log_file and has_log_file:
        assert (pkl_task.path(pkl_task.torch_log_file)
                == pkl_task.path(pkl_task.log_file)), 'mismatched log files'
    if has_log_file:
        task.set_file('log', _relative_path(pkl_task, pkl_task.log_file))
    elif has_torch_log_file:
        task.set_file('log', _relative_path(pkl_task, pkl_task.torch_log_file))

    # snapshots
    for snapshot_path, snapshot_epoch in pkl_task.snapshots:
        model = database.Model()
        model.training = training

        model.set_attribute('epoch', snapshot_epoch)
        model.set_file('weights', _relative_path(pkl_task, snapshot_path))
        model.set_file('network_architecture', _relative_path(pkl_task, inference_architecture_file))

    return task
