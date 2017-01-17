# Copyright (c) 2015-2017, NVIDIA CORPORATION.  All rights reserved.
from __future__ import absolute_import

import imp
import os
import platform
import re
import subprocess
import sys

from . import option_list
from digits import device_query
from digits.utils import parse_version


def load_from_envvar(envvar):
    """
    Load information from an installation indicated by an environment variable
    """
    value = os.environ[envvar].strip().strip("\"' ")

    if platform.system() == 'Windows':
        executable_dir = os.path.join(value, 'install', 'bin')
        python_dir = os.path.join(value, 'install', 'python')
    else:
        executable_dir = os.path.join(value, 'build', 'tools')
        python_dir = os.path.join(value, 'python')

    try:
        executable = find_executable_in_dir(executable_dir)
        if executable is None:
            raise ValueError('Caffe executable not found at "%s"'
                             % executable_dir)
        if not is_pycaffe_in_dir(python_dir):
            raise ValueError('Pycaffe not found in "%s"'
                             % python_dir)
        import_pycaffe(python_dir)
        version, flavor = get_version_and_flavor(executable)
    except:
        print ('"%s" from %s does not point to a valid installation of Caffe.'
               % (value, envvar))
        print 'Use the envvar CAFFE_ROOT to indicate a valid installation.'
        raise
    return executable, version, flavor


def load_from_path():
    """
    Load information from an installation on standard paths (PATH and PYTHONPATH)
    """
    try:
        executable = find_executable_in_dir()
        if executable is None:
            raise ValueError('Caffe executable not found in PATH')
        if not is_pycaffe_in_dir():
            raise ValueError('Pycaffe not found in PYTHONPATH')
        import_pycaffe()
        version, flavor = get_version_and_flavor(executable)
    except:
        print 'A valid Caffe installation was not found on your system.'
        print 'Use the envvar CAFFE_ROOT to indicate a valid installation.'
        raise
    return executable, version, flavor


def find_executable_in_dir(dirname=None):
    """
    Returns the path to the caffe executable at dirname
    If dirname is None, search all directories in sys.path
    Returns None if not found
    """
    if platform.system() == 'Windows':
        exe_name = 'caffe.exe'
    else:
        exe_name = 'caffe'

    if dirname is None:
        dirnames = [path.strip("\"' ") for path in os.environ['PATH'].split(os.pathsep)]
    else:
        dirnames = [dirname]

    for dirname in dirnames:
        path = os.path.join(dirname, exe_name)
        if os.path.isfile(path) and os.access(path, os.X_OK):
            return path
    return None


def is_pycaffe_in_dir(dirname=None):
    """
    Returns True if you can "import caffe" from dirname
    If dirname is None, search all directories in sys.path
    """
    old_path = sys.path
    if dirname is not None:
        sys.path = [dirname]  # temporarily replace sys.path
    try:
        imp.find_module('caffe')
    except ImportError:
        return False
    finally:
        sys.path = old_path
    return True


def import_pycaffe(dirname=None):
    """
    Imports caffe
    If dirname is not None, prepend it to sys.path first
    """
    if dirname is not None:
        sys.path.insert(0, dirname)
        # Add to PYTHONPATH so that build/tools/caffe is aware of python layers there
        os.environ['PYTHONPATH'] = '%s%s%s' % (
            dirname, os.pathsep, os.environ.get('PYTHONPATH'))

    # Suppress GLOG output for python bindings
    GLOG_minloglevel = os.environ.pop('GLOG_minloglevel', None)
    # Show only "ERROR" and "FATAL"
    os.environ['GLOG_minloglevel'] = '2'

    # for Windows environment, loading h5py before caffe solves the issue mentioned in
    # https://github.com/NVIDIA/DIGITS/issues/47#issuecomment-206292824
    import h5py  # noqa
    try:
        import caffe
    except ImportError:
        print 'Did you forget to "make pycaffe"?'
        raise

    # Strange issue with protocol buffers and pickle - see issue #32
    sys.path.insert(0, os.path.join(
        os.path.dirname(caffe.__file__), 'proto'))

    # Turn GLOG output back on for subprocess calls
    if GLOG_minloglevel is None:
        del os.environ['GLOG_minloglevel']
    else:
        os.environ['GLOG_minloglevel'] = GLOG_minloglevel


def get_version_and_flavor(executable):
    """
    Returns (version, flavor)
    Should be called after import_pycaffe()
    """
    version_string = get_version_from_pycaffe()
    if version_string is None:
        version_string = get_version_from_cmdline(executable)
    if version_string is None:
        version_string = get_version_from_soname(executable)

    if version_string is None:
        raise ValueError('Could not find version information for Caffe build ' +
                         'at "%s". Upgrade your installation' % executable)

    version = parse_version(version_string)

    if parse_version(0, 99, 0) > version > parse_version(0, 9, 0):
        flavor = 'NVIDIA'
        minimum_version = '0.11.0'
        if version < parse_version(minimum_version):
            raise ValueError(
                'Required version "%s" is greater than "%s". Upgrade your installation.'
                % (minimum_version, version_string))
    else:
        flavor = 'BVLC'

    return version_string, flavor


def get_version_from_pycaffe():
    try:
        from caffe import __version__ as version
        return version
    except ImportError:
        return None


def get_version_from_cmdline(executable):
    command = [executable, '-version']
    p = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if p.wait():
        print p.stderr.read().strip()
        raise RuntimeError('"%s" returned error code %s' % (command, p.returncode))

    pattern = 'version'
    for line in p.stdout:
        if pattern in line:
            return line[line.find(pattern) + len(pattern) + 1:].strip()
    return None


def get_version_from_soname(executable):
    command = ['ldd', executable]
    p = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if p.wait():
        print p.stderr.read().strip()
        raise RuntimeError('"%s" returned error code %s' % (command, p.returncode))

    # Search output for caffe library
    libname = 'libcaffe'
    caffe_line = None
    for line in p.stdout:
        if libname in line:
            caffe_line = line
            break

    if caffe_line is None:
        raise ValueError('libcaffe not found in linked libraries for "%s"'
                         % executable)

    # Read the symlink for libcaffe from ldd output
    symlink = caffe_line.split()[2]
    filename = os.path.basename(os.path.realpath(symlink))

    # parse the version string
    match = re.match(r'%s(.*)\.so\.(\S+)$' % (libname), filename)
    if match:
        return match.group(2)
    else:
        return None


if 'CAFFE_ROOT' in os.environ:
    executable, version, flavor = load_from_envvar('CAFFE_ROOT')
elif 'CAFFE_HOME' in os.environ:
    executable, version, flavor = load_from_envvar('CAFFE_HOME')
else:
    executable, version, flavor = load_from_path()

option_list['caffe'] = {
    'executable': executable,
    'version': version,
    'flavor': flavor,
    'multi_gpu': (flavor == 'BVLC' or parse_version(version) >= parse_version(0, 12)),
    'cuda_enabled': (len(device_query.get_devices()) > 0),
}
