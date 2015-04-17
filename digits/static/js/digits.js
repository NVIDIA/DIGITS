// Copyright (c) 2014-2015, NVIDIA CORPORATION.  All rights reserved.

function errorAlert(data) {
    var title, msg;
    if (data.status == 0) {
        title = 'An error occurred!';
        msg = '<p class="text-danger">The server may be down.</p>';
    } else {
        title = data.status + ': ' + data.statusText;
        msg = data.responseText;
    }
    bootbox.alert({
        title: title,
        message: msg,
    });
}
