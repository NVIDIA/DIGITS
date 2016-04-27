// Copyright (c) 2014-2016, NVIDIA CORPORATION.  All rights reserved.

function print_time_diff(diff) {
    if (diff < 0) {
        return 'Negative Time';
    }
    var total_seconds = Math.floor(diff);
    var days = Math.floor(total_seconds/(24*3600));
    var hours = Math.floor((total_seconds % (24*3600))/3600);
    var minutes = Math.floor((total_seconds % 3600)/60);
    var seconds = Math.floor(total_seconds % 60);

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
        return pair(days, 'day', hours, 'hour');
    else if (hours >= 1)
        return pair(hours, 'hour', minutes, 'minute');
    else if (minutes >= 1)
        return pair(minutes, 'minute', seconds, 'second');
    return plural(seconds, 'second');
}

function print_time_diff_simple(diff, min_unit) {
    if (diff == 'N/A') {
        return diff
    }
    diff = Math.max(0, diff);

    if (typeof(min_unit)==='undefined') min_unit = 'second';

    var total_seconds = Math.floor(diff);
    var days = Math.floor(total_seconds/(24*3600));
    var hours = Math.floor((total_seconds % (24*3600))/3600);
    var minutes = Math.floor((total_seconds % 3600)/60);
    var seconds = Math.floor(total_seconds % 60);

    function plural(number, name) {
        return number + ' ' + name + (number == 1 ? '' : 's');
    }

    if (days >= 1 || min_unit == 'day')
        return plural(days, 'day');
    else if (hours >= 1 || min_unit == 'hour')
        return plural(hours, 'hour');
    else if (minutes >= 1 || min_unit == 'minute')
        return plural(minutes, 'minute');
    return plural(seconds, 'second');
}

function print_time_diff_terse(diff, min_unit) {
    if (diff == 'N/A') {
        return diff
    }
    diff = Math.max(0, diff);

    if (typeof(min_unit)==='undefined') min_unit = 'second';

    var total_seconds = Math.floor(diff);
    var days = Math.floor(total_seconds/(24*3600));
    var hours = Math.floor((total_seconds % (24*3600))/3600);
    var minutes = Math.floor((total_seconds % 3600)/60);
    var seconds = Math.floor(total_seconds % 60);

    if (days >= 1 || min_unit == 'day')
        return days + 'd'
    else if (hours >= 1 || min_unit == 'hour')
        return hours + 'h'
    else if (minutes >= 1 || min_unit == 'minute')
        return minutes + 'm'
    return seconds + 's'
}

function print_time_diff_ago(start, min_unit) {
    if (start == 'N/A') {
        return start
    }

    var now = Date.now() / 1000;
    var time = print_time_diff_simple(now - start, min_unit);
    return time + ' ago'
}
