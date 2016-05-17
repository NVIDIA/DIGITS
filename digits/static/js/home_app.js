// Copyright (c) 2016, NVIDIA CORPORATION.  All rights reserved.

"use strict";

Array.prototype.contains = function(elem)
{
    return this.indexOf(elem) != -1;
}

function populate_completed_jobs() {
    $.getJSON('/completed_jobs.json', function(data) {
        // Find the dataset reference count
        var count = {};
        for (var i = 0; i < data.models.length; i++) {
            if (!count.hasOwnProperty(data.models[i].dataset_id)) {
                count[data.models[i].dataset_id] = 0;
            }
            count[data.models[i].dataset_id]++;;
        }
        for (var i = 0; i < data.datasets.length; i++) {
            if (count.hasOwnProperty(data.datasets[i].id)) {
                data.datasets[i].refs = count[data.datasets[i].id];
            } else {
                data.datasets[i].refs = 0;
            }
        }

        var scope = angular.element(document.getElementById("datasets-table")).scope();
        scope.jobs = data.datasets;
        scope.$apply();
        var scope = angular.element(document.getElementById("models-table")).scope();
        scope.jobs = data.models;
        scope.$apply();
    });
}

(function () {
    var app = angular.module('home_app', []);

    app.controller('tab_controller', function () {
        this.tab = 1;

        this.setTab = function (tabId) {
            this.tab = tabId;
        };

        this.isSet = function (tabId) {
            return this.tab === tabId;
        };
    });

    app.controller('job_controller', function($scope) {
        $scope.username = '{{username}}';
        $scope.mtime = 0;
        $scope.search_text = '';     // set the default search/filter term
        $scope.sort = {
            active: ['submitted'],
            descending: true
        }

        $scope.change_sorting = function(parameter, event) {
            var sort = $scope.sort;

            parameter = parameter.replace(' ', '_');
            parameter = parameter.toLowerCase();
            if (!event.shiftKey ||
                (sort.active.contains(parameter) && sort.active.length == 1)) {
                if (sort.active.contains(parameter)) {
                    sort.descending = !sort.descending;
                } else {
                    sort.descending = false;
                }
                sort.active = [parameter];
            } else {
                if (sort.active.contains(parameter)) {
                    var i = sort.active.indexOf(parameter);
                    sort.active.splice(i,1);
                } else {
                    sort.active.push(parameter);
                }
            }
        };

        $scope.get_icon = function(parameter) {
            var sort = $scope.sort;

            parameter = parameter.replace(' ', '_');
            parameter = parameter.toLowerCase();
            if (sort.active.contains(parameter)) {
                return sort.descending
                    ? 'glyphicon-chevron-up'
                    : 'glyphicon-chevron-down';
            }

            return '';
        }

        $scope.delete_jobs = function() {
            var trs = ts.selected;
            var job_ids = [];
            for (var i = 0; i < trs.length; i++) {
                job_ids.push(trs[i].id);
            }

            var n_job_ids = job_ids.length;
            bootbox.confirm(
                ('Are you sure you want to delete the ' +
                 (n_job_ids == 1 ? 'selected job' : n_job_ids + ' selected jobs?') +
                 '<br><br>All related files will be permanently removed.'),
                function(result) {
                    if (result)
                        $.ajax('/jobs',
                               {
                                   type: "DELETE",
                                   data: {'job_ids': job_ids},
                               })
                        .done(function() {
                            populate_completed_jobs();
                        })
                        .fail(function(data) {
                            populate_completed_jobs();
                            errorAlert(data);
                        });
                });
            return false;
        }

        $scope.print_time_diff_ago = function(start) {
            return print_time_diff_ago(start, 'minute');
        }

        $scope.print_time_diff_simple = function(diff) {
            return print_time_diff_simple(diff);
        }

        $scope.print_time_diff_terse = function(diff) {
            return print_time_diff_terse(diff);
        }

        $scope.is_today = function(date) {
            // return true if the date is from today
            var t0 = new Date(date * 1000).setHours(0, 0, 0, 0);
            var t1 = new Date().setHours(0, 0, 0, 0);
            return t0 == t1
        }

    });

    app.controller('datasets_controller', function($scope, $controller) {
        $controller('job_controller', {$scope: $scope});
        $scope.title = 'Datasets';
        $scope.fields = ['Name', 'Refs', 'Backend', 'Status', 'Elapsed', 'Submitted'];
        $scope.widths = ['', '46px', '72px', '60px', '70px', '82px'];
        $scope.ts = new TableSelection('#datasets-table');
    });

    app.controller('models_controller', function($scope, $controller) {
        $controller('job_controller', {$scope: $scope});
        $scope.title = 'Models';
        $scope.fields = ['Name', 'Framework', 'Status', 'Elapsed', 'Submitted'];
        $scope.widths = ['', '88px', '60px', '70px', '82px'];
        $scope.ts = new TableSelection('#models-table');
    });

    app.filter('positive', function ($filter) {
        return function (input) {
            return (input == 0) ? "" : input;
        }
    });

    app.filter('major_name', function ($filter) {
        return function (input) {
            return input.replace(/\s*\[.*\]/, '');
        }
    });

    app.filter('minor_name', function ($filter) {
        return function (input) {
            var match = input.match(/\s*\[(.*)\]/);
            return match ? match[0] : ''
        }
    });

    // Because jinja uses {{ and }}, tell angular to use {[ and ]}
    app.config(['$interpolateProvider', function($interpolateProvider) {
        $interpolateProvider.startSymbol('{[');
        $interpolateProvider.endSymbol(']}');
    }]);

})();

$(document).ready(function() {
    populate_completed_jobs();
});
