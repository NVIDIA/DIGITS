# Copyright (c) 2015, NVIDIA CORPORATION.  All rights reserved.

import math

import config_option
import prompt
import digits.device_query

class GpuListOption(config_option.Option):
    @staticmethod
    def config_file_key():
        return 'gpu_list'

    @classmethod
    def prompt_title(cls):
        return 'GPUs'

    @classmethod
    def prompt_message(cls):
        s = 'Attached devices:\n'
        for device_id, gpu in enumerate(digits.device_query.get_devices()):
            s += 'Device #%s:\n' % device_id
            s += '\t%-20s %s\n' % ('Name', gpu.name)
            s += '\t%-20s %s.%s\n' % ('Compute capability', gpu.major, gpu.minor)
            s += '\t%-20s %s\n' % ('Memory', cls.convert_size(gpu.totalGlobalMem))
            s += '\t%-20s %s\n' % ('Multiprocessors', gpu.multiProcessorCount)
            s += '\n'
        return s + '\nInput the IDs of the devices you would like to use, separated by commas, in order of preference.'

    def optional(self):
        return True

    def suggestions(self):
        if len(digits.device_query.get_devices()) > 0:
            return [prompt.Suggestion(
                ','.join([str(x) for x in xrange(len(digits.device_query.get_devices()))]),
                'D', desc='default', default=True)]
        else:
            return []

    @classmethod
    def visibility(self):
        if len(digits.device_query.get_devices()) == 0:
            # Nothing to see here
            return config_option.Visibility.NEVER
        else:
            return config_option.Visibility.DEFAULT

    @classmethod
    def validate(cls, value):
        if value == '':
            return value

        choices = []
        gpus = digits.device_query.get_devices()

        if not gpus:
            return ''
        if len(gpus) and not value.strip():
            raise config_option.BadValue('Empty list')
        for word in value.split(','):
            if not word:
                continue
            try:
                num = int(word)
            except ValueError as e:
                raise config_option.BadValue(e.message)

            if not 0 <= num < len(gpus):
                raise config_option.BadValue('There is no GPU #%d' % num)
            if num in choices:
                raise config_option.BadValue('You cannot select a GPU twice')
            choices.append(num)

        if len(choices) > 0:
            return ','.join(str(n) for n in choices)
        else:
            raise config_option.BadValue('Empty list')

    @classmethod
    def convert_size(cls, size):
        size_name = ("B", "KB", "MB", "GB", "TB", "PB", "EB", "ZB", "YB")
        i = int(math.floor(math.log(size,1024)))
        p = math.pow(1024,i)
        s = round(size/p,2)
        if (s > 0):
            return '%s %s' % (s,size_name[i])
        else:
            return '0B'

