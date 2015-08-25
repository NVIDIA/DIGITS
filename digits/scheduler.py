# Copyright (c) 2014-2015, NVIDIA CORPORATION.  All rights reserved.

import os
import time
import shutil
import traceback
import signal

import gevent
import gevent.event
import gevent.queue

from config import config_value
from . import utils
from status import Status
from job import Job
from dataset import DatasetJob
from model import ModelJob
from digits.utils import errors
from log import logger

class Resource(object):
    """
    Stores information about which tasks are using a resource
    """

    class ResourceAllocation(object):
        """
        Marks that a task is using [part of] a resource
        """
        def __init__(self, task, value):
            """
            Arguments:
            task -- which task is using the resource
            value -- how much of the resource is being used
            """
            self.task = task
            self.value = value

    def __init__(self, identifier=None, max_value=1):
        """
        Keyword arguments:
        identifier -- some way to identify this resource
        max_value -- a numeric representation of the capacity of this resource
        """
        if identifier is None:
            self.identifier = id(self)
        else:
            self.identifier = identifier
        self.max_value = max_value
        self.allocations = []

    def remaining(self):
        """
        Returns the amount of this resource that is not being used
        """
        return self.max_value - sum(a.value for a in self.allocations)

    def allocate(self, task, value):
        """
        A task is requesting to use this resource
        """
        if self.remaining() - value < 0:
            raise RuntimeError('Resource is already maxed out at %s/%s' % (
                self.remaining(),
                self.max_value)
                )
        self.allocations.append(self.ResourceAllocation(task, value))

    def deallocate(self, task):
        """
        The task has finished using this resource
        """
        for i, a in enumerate(self.allocations):
            if id(task) == id(a.task):
                self.allocations.pop(i)
                return True
        return False

class Scheduler:
    """
    Coordinates execution of Jobs
    """

    def __init__(self, gpu_list=None, verbose=False):
        """
        Keyword arguments:
        gpu_list -- a comma-separated string which is a list of GPU id's
        verbose -- if True, print more errors
        """
        self.jobs = []
        self.verbose = verbose

        # Keeps track of resource usage
        self.resources = {
                # TODO: break this into CPU cores, memory usage, IO usage, etc.
                'parse_folder_task_pool': [Resource()],
                'create_db_task_pool': [Resource(max_value=2)],
                'analyze_db_task_pool': [Resource(max_value=4)],
                'gpus': [Resource(identifier=index)
                    for index in gpu_list.split(',')] if gpu_list else [],
                }

        self.running = False
        self.shutdown = gevent.event.Event()

    def load_past_jobs(self):
        """
        Look in the jobs directory and load all valid jobs
        """
        failed = 0
        loaded_jobs = []
        for dir_name in sorted(os.listdir(config_value('jobs_dir'))):
            if os.path.isdir(os.path.join(config_value('jobs_dir'), dir_name)):
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

            # Need to fix this properly
            # if True or flask._app_ctx_stack.top is not None:
            from digits.webapp import app
            with app.app_context():
                # send message to job_management room that the job is added
                import flask
                html = flask.render_template('job_row.html', job = job)

                # Convert the html into a list for the jQuery
                # DataTable.row.add() method.  This regex removes the <tr>
                # and <td> tags, and splits the string into one element
                # for each cell.
                import re
                html = re.sub('<tr[^<]*>[\s\n\r]*<td[^<]*>[\s\n\r]*', '', html)
                html = re.sub('[\s\n\r]*</td>[\s\n\r]*</tr>', '', html)
                html = re.split('</td>[\s\n\r]*<td[^<]*>', html)

                from digits.webapp import socketio
                socketio.emit('job update',
                              {
                                  'update': 'added',
                                  'job_id': job.id(),
                                  'html': html
                              },
                              namespace='/jobs',
                              room='job_management',
                          )

            if 'DIGITS_MODE_TEST' not in os.environ:
                # Let the scheduler do a little work before returning
                time.sleep(utils.wait_time())
            return True

    def get_job(self, job_id):
        """
        Look through self.jobs to try to find the Job
        Returns None if not found
        """
        if job_id is None:
            return None
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
        dependent_jobs = []
        # try to find the job
        for i, job in enumerate(self.jobs):
            if job.id() == job_id:
                if isinstance(job, DatasetJob):
                    # check for dependencies
                    for j in self.jobs:
                        if isinstance(j, ModelJob) and j.dataset_id == job.id():
                            logger.error('Cannot delete "%s" (%s) because "%s" (%s) depends on it.' % (job.name(), job.id(), j.name(), j.id()))
                            dependent_jobs.append(j.name())
                if len(dependent_jobs)>0:
                    error_message = 'Cannot delete "%s" because %d model%s depend%s on it: %s' % (
                            job.name(),
                            len(dependent_jobs),
                            ('s' if len(dependent_jobs) != 1 else ''),
                            ('s' if len(dependent_jobs) == 1 else ''),
                            ', '.join(['"%s"' % j for j in dependent_jobs]))
                    raise errors.DeleteError(error_message)
                self.jobs.pop(i)
                job.abort()
                if os.path.exists(job.dir()):
                    shutil.rmtree(job.dir())
                logger.info('Job deleted.', job_id=job_id)
                from digits.webapp import socketio
                socketio.emit('job update',
                              {
                                  'update': 'deleted',
                                  'job_id': job.id()
                              },
                              namespace='/jobs',
                              room='job_management',
                )
                return True

        # see if the folder exists on disk
        path = os.path.join(config_value('jobs_dir'), job_id)
        path = os.path.normpath(path)
        if os.path.dirname(path) == config_value('jobs_dir') and os.path.exists(path):
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

    def completed_model_jobs(self):
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
                        if 'DIGITS_MODE_TEST' in os.environ:
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
                            if task.status in [Status.INIT, Status.WAIT]:
                                alldone = False
                                # try to start the task
                                if task.ready_to_queue():
                                    requested_resources = task.offer_resources(self.resources)
                                    if requested_resources is None:
                                        task.status = Status.WAIT
                                    else:
                                        if self.reserve_resources(task, requested_resources):
                                            gevent.spawn(self.run_task,
                                                    task, requested_resources)
                            elif task.status == Status.RUN:
                                # job is not done
                                alldone = False
                            elif task.status in [Status.DONE, Status.ABORT]:
                                # job is done
                                pass
                            elif task.status == Status.ERROR:
                                # propogate error status up to job
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

    def task_error(self, task, error):
        """
        Handle an error while executing a task
        """
        logger.error('%s: %s' % (type(error).__name__, error), job_id=task.job_id)
        task.exception = error
        task.traceback = traceback.format_exc()
        task.status = Status.ERROR

    def reserve_resources(self, task, resources):
        """
        Reserve resources for a task
        """
        try:
            # reserve resources
            for resource_type, requests in resources.iteritems():
                for identifier, value in requests:
                    found = False
                    for resource in self.resources[resource_type]:
                        if resource.identifier == identifier:
                            resource.allocate(task, value)
                            self.emit_gpus_available()
                            found = True
                            break
                    if not found:
                        raise RuntimeError('Resource "%s" with identifier="%s" not found' % (
                            resource_type, identifier))
            task.current_resources = resources
            return True
        except Exception as e:
            self.task_error(task, e)
            self.release_resources(task, resources)
            return False

    def release_resources(self, task, resources):
        """
        Release resources previously reserved for a task
        """
        # release resources
        for resource_type, requests in resources.iteritems():
            for identifier, value in requests:
                for resource in self.resources[resource_type]:
                    if resource.identifier == identifier:
                        resource.deallocate(task)
                        self.emit_gpus_available()
        task.current_resources = None

    def run_task(self, task, resources):
        """
        Executes a task

        Arguments:
        task -- the task to run
        resources -- the resources allocated for this task
            a dict mapping resource_type to lists of (identifier, value) tuples
        """
        try:
            task.run(resources)
        except Exception as e:
            self.task_error(task, e)
        finally:
            self.release_resources(task, resources)

    def emit_gpus_available(self):
        """
        Call socketio.emit gpu availablity
        """
        from digits.webapp import scheduler, socketio
        socketio.emit('server update',
                      {
                          'update': 'gpus_available',
                          'total_gpu_count': len(self.resources['gpus']),
                          'remaining_gpu_count': sum(r.remaining() for r in scheduler.resources['gpus']),
                      },
                      namespace='/jobs',
                      room='job_management'
                  )
