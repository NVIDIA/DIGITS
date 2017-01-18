// Copyright (c) 2014-2017, NVIDIA CORPORATION.  All rights reserved.

function print_time_diff(diff) {
    if (diff < 0) {
        return 'Negative Time';
    }
    var total_seconds = Math.floor(diff);
    var days = total_seconds / (24 * 3600);
    var hours = (total_seconds % (24 * 3600)) / 3600;
    var minutes = (total_seconds % 3600) / 60;
    var seconds = total_seconds % 60;

    function plural(number, name) {
        return number + ' ' + name + (number == 1 ? '' : 's');
    }

    function pair(number1, name1, number2, name2) {
        if (number2 > 0)
            return plural(number1, name1) + ', ' + plural(number2, name2);
        else
            return plural(number1, name1);
    }

    if (days >= 1)
        return pair(Math.floor(days), 'day', Math.round(hours), 'hour');
    else if (hours >= 1)
        return pair(Math.floor(hours), 'hour', Math.round(minutes), 'minute');
    else if (minutes >= 1)
        return pair(Math.floor(minutes), 'minute', Math.round(seconds), 'second');
    return plural(Math.round(seconds), 'second');
}

function print_time_diff_simple(diff, min_unit) {
    if (diff == 'N/A') {
        return diff;
    }
    diff = Math.max(0, diff);

    if (typeof(min_unit) === 'undefined') min_unit = 'second';

    var total_seconds = Math.floor(diff);
    var days = total_seconds / (24 * 3600);
    var hours = (total_seconds % (24 * 3600)) / 3600;
    var minutes = (total_seconds % 3600) / 60;
    var seconds = total_seconds % 60;

    function plural(number, name) {
        return number + ' ' + name + (number == 1 ? '' : 's');
    }

    if (days >= 1 || min_unit == 'day')
        return plural(Math.round(days), 'day');
    else if (hours >= 1 || min_unit == 'hour')
        return plural(Math.round(hours), 'hour');
    else if (minutes >= 1 || min_unit == 'minute')
        return plural(Math.round(minutes), 'minute');
    return plural(Math.round(seconds), 'second');
}

function print_time_diff_terse(diff, min_unit) {
    if (diff == 'N/A') {
        return diff;
    }
    diff = Math.max(0, diff);

    if (typeof(min_unit) === 'undefined') min_unit = 'second';

    var total_seconds = Math.floor(diff);
    var days = total_seconds / (24 * 3600);
    var hours = (total_seconds % (24 * 3600)) / 3600;
    var minutes = (total_seconds % 3600) / 60;
    var seconds = total_seconds % 60;

    if (days >= 1 || min_unit == 'day')
        return Math.round(days) + 'd';
    else if (hours >= 1 || min_unit == 'hour')
        return Math.round(hours) + 'h';
    else if (minutes >= 1 || min_unit == 'minute')
        return Math.round(minutes) + 'm';
    return Math.round(seconds) + 's';
}

function print_time_diff_ago(start, min_unit) {
    if (start == 'N/A') {
        return start;
    }

    var now = Date.now() / 1000;
    var time = print_time_diff_simple(now - start, min_unit);
    return time + ' ago';
}
