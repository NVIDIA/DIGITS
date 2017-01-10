
import os
import tempfile
import digits


def get_digits_tmpdir():
    # users should set DIGITS_TMP to a dir that is available to all nodes
    if os.environ.get('JENKINS_URL') is not None:
        os.environ['DIGITS_SLURM_TMP'] = os.environ.get('WORKSPACE') + "/tmp"
    if os.environ.get('DIGITS_SLURM_TMP') is None:
        os.environ['DIGITS_SLURM_TMP'] = os.environ.get('HOME') + "/tmp"
    os.environ['TMPDIR'] = os.path.abspath(os.environ.get('DIGITS_SLURM_TMP'))
    os.environ['TEMP'] = os.path.abspath(os.environ.get('DIGITS_SLURM_TMP'))
    os.environ['TMP'] = os.path.abspath(os.environ.get('DIGITS_SLURM_TMP'))
    tempfile.tempdir = os.environ['TMPDIR']
    return os.environ['TMPDIR']


def test_if_slurm_system():
    try:
        if os.environ.get('SLURM_HOME'):
            # get_digits_tmpdir()
            return True
        else:
            return False

    except OSError:
        return False

class slurm_manager:
    def __init__(self):
        get_digits_tmpdir()
        pass

    def pack_args(self,args, time_limit, cpu_count, mem, gpu_count, t_type):
        gpu_arg_idx = [i for i, arg in enumerate(args) if arg.startswith('--gpu')]
        if gpu_arg_idx:
            gpu_arg_idx = gpu_arg_idx[0]
        gpus = gpu_count
        print time_limit
        print cpu_count
        print mem

        if not time_limit or time_limit == 0:
            time_limit = 70
        if not cpu_count:
            cpu_count = 4
        if not mem:
            mem = 8



        # set caffe to use all available gpus
        # This is assuming that $CUDA_VISIBLE_DEVICES is set for each task on the nodes\


        if issubclass(t_type,digits.model.tasks.TrainTask):
            if gpu_arg_idx:
                args[gpu_arg_idx] = '--gpu=all'

        if gpus == 0:
            args = ['salloc', '-t', str(time_limit), '-c', str(cpu_count),
                    '--mem=' + str(mem) + 'GB', 'srun'] + args
        else:
            args = ['salloc', '-t', str(time_limit), '-c', str(cpu_count),
                    '--mem=' + str(mem) + 'GB',
                    '--gres=gpu:' + str(gpus), 'srun'] + args

        return args