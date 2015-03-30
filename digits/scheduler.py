# Copyright (c) 2014-2015, NVIDIA CORPORATION.  All rights reserved.

import os
import time
import shutil
import traceback
import signal

import gevent
import gevent.event
import gevent.queue

from config import config_option
from . import utils
from status import Status
from job import Job
from dataset import DatasetJob, tasks as dataset_tasks
from model import ModelJob, tasks as model_tasks

from log import logger

class Scheduler:
    """
    Coordinates execution of Jobs
    """
    # How many concurrent tasks will run
    NUM_SPLIT_THREADS = 1
    NUM_CREATE_THREADS = 2

    def __init__(self, gpu_list, verbose=False):
        """
        Arguments:
        gpu_list -- a list of GPUs to be distributed for training tasks

        Keyword arguments:
        verbose -- if True, print more errors
        """
        self.jobs = []

        # Keeps track of which GPUs are in use
        self.gpu_list = []
        if gpu_list and gpu_list != 'NONE':
            if isinstance(gpu_list, str):
                gpu_list = [int(x) for x in gpu_list.split(',')]
            elif isinstance(gpu_list, list):
                pass
            else:
                raise ValueError('invalid gpu_list: %s' % gpu_list)
            for index in gpu_list:
                self.gpu_list.append({'index': int(index), 'active': False})

        self.verbose = verbose

        self.split_queue = gevent.queue.Queue()
        self.create_queue = gevent.queue.Queue()
        self.train_queue = gevent.queue.Queue()

        self.running = False
        self.shutdown = gevent.event.Event()

    def load_past_jobs(self):
        """
        Look in the jobs directory and load all valid jobs
        """
        failed = 0
        loaded_jobs = []
        for dir_name in sorted(os.listdir(config_option('jobs_dir'))):
            if os.path.isdir(os.path.join(config_option('jobs_dir'), dir_name)):
                exists = False

                # Make sure it hasn't already been loaded
                for job in self.jobs:
                    if job.id() == dir_name:
                        exists = True
                        break

                if not exists:
                    try:
                        job = Job.load(dir_name)
                        # The server might have crashed
                        if job.status.is_running():
                            job.status = Status.ABORT
                        for task in job.tasks:
                            if task.status.is_running():
                                task.status = Status.ABORT

                        # We might have changed some attributes here or in __setstate__
                        job.save()
                        loaded_jobs.append(job)
                    except Exception as e:
                        failed += 1
                        if self.verbose:
                            if str(e):
                                print 'Caught %s while loading job "%s":' % (type(e).__name__, dir_name)
                                print '\t%s' % e
                            else:
                                print 'Caught %s while loading job "%s"' % (type(e).__name__, dir_name)

        # add DatasetJobs
        for job in loaded_jobs:
            if isinstance(job, DatasetJob):
                self.jobs.append(job)

        # add ModelJobs
        for job in loaded_jobs:
            if isinstance(job, ModelJob):
                try:
                    # load the DatasetJob
                    job.load_dataset()
                    self.jobs.append(job)
                except Exception as e:
                    failed += 1
                    if self.verbose:
                        if str(e):
                            print 'Caught %s while loading job "%s":' % (type(e).__name__, job.id())
                            print '\t%s' % e
                        else:
                            print 'Caught %s while loading job "%s"' % (type(e).__name__, job.id())

        if failed > 0 and self.verbose:
            print 'WARNING:', failed, 'jobs failed to load.'

    def add_job(self, job):
        """
        Add a job to self.jobs
        """
        if not self.running:
            logger.error('Scheduler not running. Cannot add job.')
            return False
        else:
            self.jobs.append(job)
            # Let the scheduler do a little work before returning
            time.sleep(utils.wait_time())
            return True

    def get_job(self, job_id):
        """
        Look through self.jobs to try to find the Job
        Returns None if not found
        """
        for j in self.jobs:
            if j.id() == job_id:
                return j
        return None

    def abort_job(self, job_id):
        """
        Aborts a running Job
        Returns True if the job was found and aborted
        """
        job = self.get_job(job_id)
        if job is None or not job.status.is_running():
            return False

        job.abort()
        return True

    def delete_job(self, job):
        """
        Deletes an entire job folder from disk
        Returns True if the Job was found and deleted
        """
        if isinstance(job, str) or isinstance(job, unicode):
            job_id = str(job)
        elif isinstance(job, Job):
            job_id = job.id()
        else:
            raise ValueError('called delete_job with a %s' % type(job))

        # try to find the job
        for i, job in enumerate(self.jobs):
            if job.id() == job_id:
                if isinstance(job, DatasetJob):
                    # check for dependencies
                    for j in self.jobs:
                        if isinstance(j, ModelJob) and j.dataset_id == job.id():
                            logger.error('Cannot delete %s (%s) because %s (%s) depends on it.' % (job.name(), job.id(), j.name(), j.id()))
                            return False
                self.jobs.pop(i)
                job.abort()
                if os.path.exists(job.dir()):
                    shutil.rmtree(job.dir())
                logger.info('Job deleted.', job_id=job_id)
                return True

        # see if the folder exists on disk
        path = os.path.join(config_option('jobs_dir'), job_id)
        path = os.path.normpath(path)
        if os.path.dirname(path) == config_option('jobs_dir') and os.path.exists(path):
            shutil.rmtree(path)
            return True

        return False

    def running_dataset_jobs(self):
        """a query utility"""
        return sorted(
                [j for j in self.jobs if isinstance(j, DatasetJob) and j.status.is_running()],
                cmp=lambda x,y: cmp(y.id(), x.id())
                )

    def completed_dataset_jobs(self):
        """a query utility"""
        return sorted(
                [j for j in self.jobs if isinstance(j, DatasetJob) and not j.status.is_running()],
                cmp=lambda x,y: cmp(y.id(), x.id())
                )

    def running_model_jobs(self):
        """a query utility"""
        return sorted(
                [j for j in self.jobs if isinstance(j, ModelJob) and j.status.is_running()],
                cmp=lambda x,y: cmp(y.id(), x.id())
                )

    def running_model_jobs(self):
        """a query utility"""
        return sorted(
                [j for j in self.jobs if isinstance(j, ModelJob) and not j.status.is_running()],
                cmp=lambda x,y: cmp(y.id(), x.id())
                )

    def start(self):
        """
        Start the Scheduler
        Returns True on success
        """
        if self.running:
            return True

        gevent.spawn(self.main_thread)

        for x in xrange(self.NUM_SPLIT_THREADS):
            gevent.spawn(self.task_thread, self.split_queue)

        for x in xrange(self.NUM_CREATE_THREADS):
            gevent.spawn(self.task_thread, self.create_queue)

        if len(self.gpu_list):
            for x in xrange(len(self.gpu_list)):
                gevent.spawn(self.task_thread, self.train_queue)
        else:
            for x in xrange(1): # Only start 1 thread if running in CPU mode
                gevent.spawn(self.task_thread, self.train_queue)

        self.running = True
        return True

    def stop(self):
        """
        Stop the Scheduler
        Returns True if the shutdown was graceful
        """
        self.shutdown.set()
        wait_limit = 5
        start = time.time()
        while self.running:
            if time.time() - start > wait_limit:
                return False
            time.sleep(0.1)
        return True

    def main_thread(self):
        """
        Monitors the jobs in current_jobs, updates their statuses,
        and puts their tasks in queues to be processed by other threads
        """
        signal.signal(signal.SIGTERM, self.sigterm_handler)
        try:
            last_saved = None
            while not self.shutdown.is_set():
                # Iterate backwards so we can delete jobs
                for job in reversed(self.jobs):
                    if job.status == Status.INIT:
                        def start_this_job(job):
                            if isinstance(job, ModelJob):
                                if job.dataset.status == Status.DONE:
                                    job.status = Status.RUN
                                elif job.dataset.status in [Status.ABORT, Status.ERROR]:
                                    job.abort()
                                else:
                                    job.status = Status.WAIT
                            else:
                                job.status = Status.RUN
                        if config_option('level') == 'test':
                            start_this_job(job)
                        else:
                            # Delay start by one second for initial page load
                            gevent.spawn_later(1, start_this_job, job)

                    if job.status == Status.WAIT:
                        if isinstance(job, ModelJob):
                            if job.dataset.status == Status.DONE:
                                job.status = Status.RUN
                            elif job.dataset.status in [Status.ABORT, Status.ERROR]:
                                job.abort()
                        else:
                            job.status = Status.RUN

                    if job.status == Status.RUN:
                        alldone = True
                        for task in job.tasks:
                            if task.status == Status.INIT:
                                alldone = False
                                if task.ready_to_queue():
                                    logger.debug('%s task queued.' % task.name(), job_id=job.id())
                                    task.status = Status.WAIT
                                    if isinstance(task, dataset_tasks.ParseFolderTask):
                                        self.split_queue.put( (job, task) )
                                    elif isinstance(task, dataset_tasks.CreateDbTask):
                                        self.create_queue.put( (job, task) )
                                    elif isinstance(task, model_tasks.TrainTask):
                                        self.train_queue.put( (job, task) )
                                    else:
                                        logger.error('Task type %s not recognized' % type(task).__name__, job_id=job.id())
                                        task.exception = Exception('Task type not recognized')
                                        task.status = Status.ERROR
                            elif task.status == Status.WAIT or task.status == Status.RUN:
                                alldone = False
                            elif task.status == Status.DONE:
                                pass
                            elif task.status == Status.ABORT:
                                pass
                            elif task.status == Status.ERROR:
                                job.status = Status.ERROR
                                alldone = False
                                break
                            else:
                                logger.warning('Unrecognized task status: "%s"', task.status, job_id=job.id())
                        if alldone:
                            job.status = Status.DONE
                            logger.info('Job complete.', job_id=job.id())
                            job.save()

                # save running jobs every 15 seconds
                if not last_saved or time.time()-last_saved > 15:
                    for job in self.jobs:
                        if job.status.is_running():
                            job.save()
                    last_saved = time.time()

                time.sleep(utils.wait_time())
        except KeyboardInterrupt:
            pass

        # Shutdown
        for job in self.jobs:
            job.abort()
            job.save()
        self.running = False

    def sigterm_handler(self, signal, frame):
        """
        Gunicorn shuts down workers with SIGTERM, not SIGKILL
        """
        self.shutdown.set()

    def task_thread(self, queue):
        """
        Executes tasks in queue
        """
        while not self.shutdown.is_set():
            if queue.empty() is False:
                (job, task) = queue.get_nowait()

                # Don't run the task if the job is done
                if job.status in [Status.ERROR, Status.ABORT]:
                    task.status = Status.ABORT
                else:
                    options = {}
                    gpu_id = -1
                    try:
                        if isinstance(task, model_tasks.TrainTask):
                            ### Select GPU
                            if len(self.gpu_list):
                                for gpu in self.gpu_list:
                                    if not gpu['active']:
                                        gpu_id = gpu['index']
                                        gpu['active'] = True
                                        break
                                assert gpu_id != -1, 'no available GPU'
                            else:
                                gpu_id = None
                            options['gpu_id'] = gpu_id

                        task.run(**options)

                    except Exception as e:
                        logger.error('%s: %s' % (type(e).__name__, e), job_id=job.id())
                        task.exception = e
                        task.traceback = traceback.format_exc()
                        task.status = Status.ERROR
                    finally:
                        ### Release GPU
                        if gpu_id != -1 and gpu_id is not None:
                            for gpu in self.gpu_list:
                                if gpu['index'] == gpu_id:
                                    gpu['active'] = False
            else:
                # Wait before checking again for a task
                time.sleep(utils.wait_time())

