# Copyright (c) 2014-2015, NVIDIA CORPORATION.  All rights reserved.

import time

def print_time(t):
    lt = time.localtime(t)
    now = time.localtime()

    if lt.tm_year != now.tm_year:
        return time.strftime('%b %d %Y, %I:%M:%S %p', lt)
    elif lt.tm_mon != now.tm_mon:
        return time.strftime('%b %d, %I:%M:%S %p', lt)
    elif lt.tm_mday != now.tm_mday:
        return time.strftime('%a %b %d, %I:%M:%S %p', lt)
    else:
        return time.strftime('%I:%M:%S %p', lt)

def print_time_diff(diff):
    if diff is None:
        return '?'

    total_seconds = int(diff)
    days = total_seconds//(24*3600)
    hours = (total_seconds % (24*3600))//3600
    minutes = (total_seconds % 3600)//60
    seconds = total_seconds % 60
    if days > 1:
        return '%d days, %d hours' % (days, hours)
    elif days == 1:
        return '1 day, %d hours' % hours
    elif hours > 1:
        return '%d hours, %d minutes' % (hours, minutes)
    elif hours == 1:
        return '1 hour, %d minutes' % minutes
    elif minutes > 1:
        return '%d minutes, %d seconds' % (minutes, seconds)
    elif minutes == 1:
        return '1 minute, %d seconds' % seconds
    elif seconds == 1:
        return '1 second'
    return '%s seconds' % seconds

def print_time_since(t):
    return print_time_diff(time.time() - t)
