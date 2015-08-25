// Copyright (c) 2014-2015, NVIDIA CORPORATION.  All rights reserved.

function print_time_diff(diff) {
    if (diff < 0) {
        return 'Negative Time';
    }

    var total_seconds = Math.round(diff);
    var days = Math.round(total_seconds/(24*3600));
    var hours = Math.round((total_seconds % (24*3600))/3600);
    var minutes = Math.round((total_seconds % 3600)/60);
    var seconds = Math.round(total_seconds % 60);

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
