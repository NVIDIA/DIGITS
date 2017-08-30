// Copyright (c) 2016-2017, NVIDIA CORPORATION.  All rights reserved.

'use strict';

try {
(function() {
    var app = angular.module('home_app', ['ngStorage', 'ui.bootstrap'])
        .filter('html', function($sce) {
        return function(input) {
            return $sce.trustAsHtml(input);
        }
    });

    app.controller('tab_controller', function($scope) {
        var self = this;
        $scope.init = function(tab) {
          self.tab = _.isUndefined(tab) ? 2 : tab;
        };

        this.setTab = function(tabId, $event) {
            this.tab = tabId;
            $event.stopPropagation();
        };

        this.isSet = function(tabId) {
            return this.tab === tabId;
        };
    });

    app.controller('all_jobs_controller', function($rootScope, $scope, $localStorage, $http) {
        $scope.jobs = [];

        this.storage = $localStorage.$default({
            show_groups: true,
        });

        $scope.add_job = function(job_id) {
            $http({
                method: 'GET',
                url: URL_PREFIX + '/jobs/' + job_id + '/table_data.json',
            }).then(function success(response) {
                var job = response.data.job;
                for (var i = 0; i < $scope.jobs.length; i++) {
                    if ($scope.jobs[i].id == job_id) {
                        $scope.jobs[i] = Object.assign({}, job);
                        return;
                    }
                }
                $scope.jobs.push(job);
            });
        };

        $scope.remove_job = function(job_id) {
            for (var i = 0; i < $scope.jobs.length; i++) {
                if ($scope.jobs[i].id == job_id) {
                    $scope.jobs.splice(i, 1);
                    return true;
                }
            }
            return false;
        };

        $scope.deselect_all = function() {
            for (var i = 0; i < $scope.jobs.length; i++) {
                $scope.jobs[i].selected = false;
            }
        };

        $scope.load_jobs = function() {
            $http({
                method: 'GET',
                url: URL_PREFIX + '/completed_jobs.json',
            }).then(function success(response) {
                // Find the dataset reference count
                var count = {};
                for (var i = 0; i < response.data.models.length; i++) {
                    if (!count.hasOwnProperty(response.data.models[i].dataset_id)) {
                        count[response.data.models[i].dataset_id] = 0;
                    }
                    count[response.data.models[i].dataset_id]++;
                }
                for (var i = 0; i < response.data.datasets.length; i++) {
                    if (count.hasOwnProperty(response.data.datasets[i].id)) {
                        response.data.datasets[i].refs = count[response.data.datasets[i].id];
                    } else {
                        response.data.datasets[i].refs = 0;
                    }
                }


                var r = response.data;
                $scope.jobs = [].concat(r.running, r.datasets, r.models, r.pretrained_models);

                var scope = angular.element(document.getElementById('models-table')).scope();
                // scope.storage.model_output_fields = [];
                for (var i = 0; i < response.data.model_output_fields.length; i++) {
                    var found = false;
                    for (var j = 0; j < scope.storage.model_output_fields.length; j++) {
                        if (response.data.model_output_fields[i] == scope.storage.model_output_fields[j].name) {
                            found = true;
                            break;
                        }
                    }
                    if (!found) {
                        scope.storage.model_output_fields.push({
                        'name': response.data.model_output_fields[i],
                        'show': false});
                    }
                }
            }, function error(response) {
                console.log(response.statusText);
            });
        };
        $scope.load_jobs();

        $scope.is_running = function(job) {
            return (job &&
                    (job.status == 'Initialized' ||
                     job.status == 'Waiting' ||
                     job.status == 'Running'));
        };

        $scope.is_dataset = function(job) {
            return (job &&
                    (!$scope.is_running(job) &&
                     job.type == 'dataset'));
        };

        $scope.is_model = function(job) {
            return (!$scope.is_running(job) && job.type == 'model');
        };

        $scope.is_pretrained_model = function(job) {
            return (!$scope.is_running(job) && job.type == 'pretrained_model');
        };

        $scope.set_attribute = function(job_id, name, value) {
            for (var i = 0; i < $scope.jobs.length; i++) {
                if ($scope.jobs[i].id == job_id) {
                    $scope.jobs[i][name] = value;
                    return true;
                }
            }
            return false;
        };

        $scope.print = function(txt) {
            console.log(txt);
        };
    });

    app.controller('select_controller', function($scope) {
        var last_selected_row = null;
        var mousedown_row = null;
        var mousedown_with_alt = false;
        var mousedown_with_shift = false;
        var mousedown_was_selected = false;

        function select($index) {
            if (0 <= $index && $index < $scope.jobs.length)
                $scope.jobs[$index].selected = true;
        }

        function deselect($index) {
            if (0 <= $index && $index < $scope.jobs.length)
                $scope.jobs[$index].selected = false;
        }

        function swap_select($index) {
            if (0 <= $index && $index < $scope.jobs.length)
                $scope.jobs[$index].selected = !$scope.jobs[$index].selected;
        }

        function do_selection($index) {
            var first = $index;
            var last = $index;

            if (mousedown_with_shift && last_selected_row !== null) {
                first = Math.min(last_selected_row, first);
                last = Math.max(last_selected_row, last);
            } else if (mousedown_row !== null) {
                first = Math.min(mousedown_row, first);
                last = Math.max(mousedown_row, last);
            }

            // Do we want to add, remove, or replace the selection
            var action = (mousedown_with_shift ? 'add' :
                          mousedown_with_alt ? (
                              mousedown_was_selected ? 'remove' : 'add') :
                          'replace');

            if (action == 'remove') {
                for (var i = first; i <= last; i++)
                    deselect(i);

                for (var i = 0; i < $scope.jobs.length; i++)
                    if ($scope.jobs[i].selected)
                        last_selected_row = i;
            } else {
                if (action == 'replace')
                    for (var i = 0; i < $scope.jobs.length; i++)
                        deselect(i);

                for (var i = first; i <= last; i++)
                    select(i);

                last_selected_row = $index;
            }
        }

        $scope.mousedown = function($index, $event) {
            if ($event.which != 1)
                return;

            mousedown_with_alt = ($event.ctrlKey || $event.metaKey);
            mousedown_with_shift = ($event.shiftKey);
            mousedown_was_selected = $scope.jobs[$index].selected;
            mousedown_row = null;
            do_selection($index);
            mousedown_row = $index;
            $event.stopPropagation();
        };

        // drag selection
        $scope.mousemove = function($index, $event) {
            if ($event.which != 1 || mousedown_row === null)
                return;

            $event.stopPropagation();
            if (last_selected_row === $index)
                return;

            do_selection($index);
        };

        // end drag selection
        $scope.mouseup = function($index, $event) {
            if ($event.which != 1)
                return;
            $event.stopPropagation();
            if (mousedown_row === null)
                return;
            mousedown_row = null;
        };

        $scope.click = function($index, $event) {
            if ($event.which != 1)
                return;
            $event.stopPropagation();
        };

        // Because the key event is global, only register once.
        $scope.keydown = function($event) {
            // Use alt-a for select all rows
            if (($event.ctrlKey || $event.metaKey) && $event.keyCode == 65) {
                $event.preventDefault();
                for (var i = 0; i < $scope.jobs.length; i++)
                    select(i);
                last_selected_row = null;
            }
            // Use alt-d for deselect all rows
            if (($event.ctrlKey || $event.metaKey) && $event.keyCode == 68) {
                $event.preventDefault();
                for (var i = 0; i < $scope.jobs.length; i++)
                    deselect(i);
                last_selected_row = null;
            }
            // Use alt-i for invert selection
            if (($event.ctrlKey || $event.metaKey) && $event.keyCode == 73) {
                $event.preventDefault();
                for (var i = 0; i < $scope.jobs.length; i++) {
                    swap_select(i);
                    if ($scope.jobs[i].selected)
                        last_selected_row = i;
                }
            }
            switch ($event.which) {
            case 38: // up
                if (last_selected_row !== null) {
                    if (!$event.shiftKey)
                        for (var i = 0; i < $scope.jobs.length; i++)
                            deselect(i);
                    last_selected_row = Math.max(last_selected_row - 1, 0);
                    select(last_selected_row);
                }
                break;
            case 40: // down
                if (last_selected_row !== null) {
                    if (!$event.shiftKey)
                        for (var i = 0; i < $scope.jobs.length; i++)
                            deselect(i);
                    last_selected_row = Math.min(last_selected_row + 1, $scope.jobs.length);
                    select(last_selected_row);
                }
                break;
            default: return; // exit this handler for other keys
            }
            $event.preventDefault(); // prevent the default action (scroll / move caret)
        };

        $scope.mouseleave = function($event) {
            mousedown_row = null;
        };

        $scope.any_selected = function() {
            if ($scope.jobs === undefined)
                return false;

            for (var i = 0; i < $scope.jobs.length; i++)
                if ($scope.jobs[i].selected)
                    return true;
            return false;
        };
    });

    app.controller('job_controller', function($scope, $controller) {
        $controller('select_controller', {$scope: $scope});
        $scope.username = '{{username}}';
        $scope.search_text = '';     // set the default search/filter term
        $scope.sort = {
            active1: 'submitted',
            descending1: 1,
            active2: 'submitted',
            descending2: 1,
        };
        $scope.jobs = [];

        $scope.default_descending = function(active) {
            for (var i = 0; i < $scope.jobs.length; i++) {
                if (typeof($scope.jobs[i][active]) == 'string') {
                    return -1;
                }
            }
            return 1;
        };

        $scope.change_sorting = function(parameter, event) {
            var sort = $scope.sort;
            if (!event.shiftKey || sort.active1 == parameter) {
                if (sort.active1 == parameter) {
                    sort.descending1 = sort.descending2 = -sort.descending1;
                } else {
                    sort.descending1 = sort.descending2 = $scope.default_descending(parameter);
                }
                sort.active1 = sort.active2 = parameter;
            } else {
                if (sort.active2 == parameter) {
                    sort.descending2 = -sort.descending2;
                } else {
                    sort.descending2 = $scope.default_descending(parameter);
                }
                sort.active2 = parameter;
            }
        };

        $scope.get_icon = function(parameter) {
            var sort = $scope.sort;

            if (sort.active1 == parameter) {
                return sort.descending1 == 1 ?
                    'glyphicon-chevron-up' :
                    'glyphicon-chevron-down';
            }
            if (sort.active2 == parameter) {
                return sort.descending2 == 1 ?
                    'glyphicon-chevron-up' :
                    'glyphicon-chevron-down';
            }
            return '';
        };

        $scope.get_selected_job_ids = function() {
            var job_ids = [];
            for (var i = 0; i < $scope.jobs.length; i++)
                if ($scope.jobs[i].selected)
                    job_ids.push($scope.jobs[i].id);
            return job_ids;
        };

        $scope.get_group_for_job = function(job_id) {
            for (var i = 0; i < $scope.jobs.length; i++)
                if ($scope.jobs[i].id == job_id)
                    return $scope.jobs[i].group ? $scope.jobs[i].group : '';
            return '';
        };

        $scope.delete_jobs = function() {
            var job_ids = $scope.get_selected_job_ids();
            if (job_ids.length == 0) return;
            bootbox.confirm(
                ('Are you sure you want to delete the selected ' +
                 (job_ids.length == 1 ? 'job?' : job_ids.length + ' jobs?') +
                 '<br><br>All related files will be permanently removed.'),
                function(result) {
                    if (result)
                        $.ajax(URL_PREFIX + '/jobs',
                               {
                                   type: 'DELETE',
                                   data: {'job_ids': job_ids},
                               })
                        .done(function() {
                            var scope = angular.element(document.getElementById('all-jobs')).scope();
                            scope.load_jobs();
                        })
                        .fail(function(data) {
                            var scope = angular.element(document.getElementById('all-jobs')).scope();
                            scope.load_jobs();
                            errorAlert(data);
                        });
                });
            return false;
        };

        $scope.abort_jobs = function() {
            var job_ids = $scope.get_selected_job_ids();
            if (job_ids.length == 0) return;
            bootbox.confirm(
                ('Are you sure you want to abort the selected ' +
                 (job_ids.length == 1 ? 'job?' : job_ids.length + ' jobs?')),
                function(result) {
                    if (result)
                        $.ajax(URL_PREFIX + '/abort_jobs',
                               {
                                   type: 'POST',
                                   data: {'job_ids': job_ids},
                               })
                        .done(function() {
                            var scope = angular.element(document.getElementById('all-jobs')).scope();
                            scope.load_jobs();
                        })
                        .fail(function(data) {
                            var scope = angular.element(document.getElementById('all-jobs')).scope();
                            scope.load_jobs();
                            errorAlert(data);
                        });
                });
            return false;
        };

        $scope.group_jobs = function() {
            var job_ids = $scope.get_selected_job_ids();
            if (job_ids.length == 0) return;
            var job_string = (job_ids.length == 1 ? 'job' : 'jobs');
            var n_job_string = (job_ids.length == 1 ? '' : job_ids.length);
            var default_group_name = $scope.get_group_for_job(job_ids[0]);
            bootbox.prompt({
                title: ('Enter a group name for the ' +
                        n_job_string +
                        ' selected ' +
                        job_string + '.' +
                        '<br><small>Leave the name blank to ungroup the ' +
                        job_string + '.</small>'),
                value: default_group_name,
                callback: function(result) {
                    if (result !== null) {
                        // Case the user enters 'Ungrouped', change it to ''
                        if (result == 'Ungrouped')
                            result = '';
                        $.ajax(URL_PREFIX + '/group',
                               {
                                   type: 'POST',
                                   data: {
                                       'job_ids': job_ids,
                                       group_name: result,
                                   },
                               })
                            .done(function() {
                                $scope.$apply();
                            })
                            .fail(function(data) {
                                $scope.$apply();
                                errorAlert(data);
                            });
                    }
                }
            });
            return false;
        };


        $scope.print_time_diff_ago = function(start) {
            return print_time_diff_ago(start, 'minute');
        };

        $scope.print_time_diff_simple = function(diff) {
            return print_time_diff_simple(diff);
        };

        $scope.print_time_diff_terse = function(diff) {
            return print_time_diff_terse(diff);
        };

        $scope.print_time_diff = function(diff) {
            return print_time_diff(diff);
        };

        $scope.is_today = function(date) {
            // return true if the date is from today
            var t0 = new Date(date * 1000).setHours(0, 0, 0, 0);
            var t1 = new Date().setHours(0, 0, 0, 0);
            return t0 == t1;
        };

        $scope.show = function(show) {
            return function(item) {
                return item.show === show;
            };
        };

        $scope.visualLength = function(txt, min_width)
        {
            if (min_width == undefined)
                min_width = 0;
            var ruler = document.getElementById('ruler');
            ruler.innerHTML = txt;
            var icon_width = 14;
            var width = ruler.offsetWidth + icon_width + 4;
            return Math.max(width, min_width);
        };
    });

    app.controller('running_controller', function($scope, $controller) {
        $controller('job_controller', {$scope: $scope});
        $scope.title = 'Running Jobs';
        $scope.fields = [{name: 'name', show: true, min_width: 100},
                         {name: 'submitted', show: true, min_width: 100},
                         {name: 'status', show: true, min_width: 120},
                         {name: 'loss', show: true, min_width: 200},
                         {name: 'progress', show: true, min_width: 200}];
    });

    app.controller('datasets_controller', function($scope, $controller) {
        $controller('job_controller', {$scope: $scope});
        $scope.title = 'Datasets';
        $scope.fields = [{name: 'name', show: true},
                         {name: 'refs', show: true},
                         {name: 'extension', show: true, min_width: 150},
                         {name: 'backend', show: true},
                         {name: 'status', show: true},
                         {name: 'elapsed', show: true},
                         {name: 'submitted', show: true}];
    });

    app.controller('models_controller', function($scope, $localStorage, $controller) {
        $controller('job_controller', {$scope: $scope});
        $scope.title = 'Models';
        var model_fields = [
            {name: 'name', show: true, min_width: 100},
            {name: 'id', show: false, min_width: 200},
            {name: 'extension', show: true, min_width: 150},
            {name: 'framework', show: true, min_width: 50},
            {name: 'status', show: true, min_width: 50},
            {name: 'elapsed', show: true, min_width: 50},
            {name: 'submitted', show: true, min_width: 50}
        ];
        if ($localStorage.model_fields) {
            for (var i = 0; i < model_fields.length; i++) {
                var index = $localStorage.model_fields.findIndex(
                    function(item) {
                        return item.name == model_fields[i].name;
                    });
                if (index > -1) {
                    model_fields[i] = $localStorage.model_fields[index];
                    for (var attr in $localStorage.model_fields[index]) {
                        if (model_fields[i].hasOwnProperty(attr))
                            model_fields[i][attr] = $localStorage.model_fields[index][attr];
                    }
                }
            }
        }
        $scope.storage = $localStorage.$default({
            model_output_fields: [],
            model_fields: model_fields,
        });
        $scope.storage.model_fields = model_fields.slice();
    });

    app.controller('pretrained_models_controller', function($scope, $localStorage, $controller) {
        $controller('job_controller', {$scope: $scope});
        $scope.title = 'Models';
        $scope.fields = [{name: 'name', show: true, min_width: 0},
                         {name: 'framework', show: true, min_width: 0},
                         {name: 'username', show: true, min_width: 0},
                         {name: 'has_labels', show: true, min_width: 0},
                         {name: 'status', show: true, min_width: 0},
                         {name: 'elapsed', show: true, min_width: 0},
                         {name: 'submitted', show: true, min_width: 0}];
    });

    function precision(input, sigfigs) {
        if (isNaN(input)) return input;
        if (input == 0) return input;
        var n = Math.floor(Math.log10(input)) + 1;
        n = Math.min(n, 0);
        var places = sigfigs - n;
        var factor = '1' + Array(+(places > 0 && places + 1)).join('0');
        return Math.round(input * factor) / factor;
    }

    app.filter('precision', function($filter) {
        return function(input, sigfigs) {
            return precision(input, sigfigs);
        }
    });

    app.filter('positive', function($filter) {
        return function(input) {
            return (input == 0) ? '' : input;
        }
    });

    app.filter('major_name', function($filter) {
        return function(input) {
            return input.replace(/(\w+:[\.\w]+[, \w:\.]*)$/, '');
        }
    });

    app.filter('minor_name', function($filter) {
        return function(input) {
            var match = input.match(/(\w+:[\.\w]+[, \w:\.]*)$/);
            return match ? match[0] : '';
        }
    });

    app.filter('sort_with_empty_at_end', function() {
        return function(array, scope, show_groups) {
            if (!angular.isArray(array)) return;
            array.sort(
                function(x, y)
                {
                    var xg = x.group;
                    var yg = y.group;
                    var x1 = x[scope.sort.active1];
                    var y1 = y[scope.sort.active1];
                    var x2 = x[scope.sort.active2];
                    var y2 = y[scope.sort.active2];
                    var d1 = scope.sort.descending1;
                    var d2 = scope.sort.descending2;

                    if (show_groups && xg != yg) {
                        if (xg === '') return 1;
                        if (yg === '') return -1;
                        return ((xg < yg) ? -1 : (xg > yg) ? 1 : 0);
                    }
                    if (x1 != y1) {
                        if (x1 === undefined) return 1;
                        if (y1 === undefined) return -1;
                        return ((x1 < y1) ? d1 : (x1 > y1) ? -d1 : 0);
                    }
                    if (x2 === undefined) return 1;
                    if (y2 === undefined) return -1;
                    return ((x2 < y2) ? d2 : (x2 > y2) ? -d2 : 0);
                }
            );

            return array;
        };
    });

    app.directive('a', function() {
        return {
            restrict: 'E',
            link: function(scope, elem, attrs) {
                if (attrs.href !== '' && attrs.href !== '#') {
                    elem.on('mousedown', function($event) {
                        $event.stopPropagation();
                    });
                    elem.on('mousemove', function($event) {
                        $event.stopPropagation();
                    });
                    elem.on('mouseup', function($event) {
                        $event.stopPropagation();
                    });
                    elem.on('click', function($event) {
                        $event.stopPropagation();
                    });
                }
            }
        };
    });

    app.directive('sparkline', function() {
        return {
            restrict: 'E',
            scope: {
                data: '@'
            },
            compile: function(tElement, tAttrs, transclude) {
                tElement.replaceWith('<span>' + tAttrs.data + '</span>');
                return function(scope, element, attrs) {
                    attrs.$observe('data', function(data) {
                        data = JSON.parse(data);
                        if (typeof(data) == 'undefined') {
                            console.info('bad sparkline');
                            element.html('');
                            return;
                        }
                        if (data.length < 2) {
                            element.html('');
                            return;
                        }
                        element.sparkline(data,
                                          {width: '150px',
                                           height: '20px',
                                           fillColor: '#c0d0f0',
                                           lineColor: '#0000f0',
                                           spotColor: false,
                                           //chartRangeMin: 0,
                                           //chartRangeMax: 100,
                                          });
                    });
                };
            }
        };
    });

    app.directive('dgName', function() {
        return {
            restrict: 'AE',
            replace: true,
            template: ('<span>' +
                        '    <a href="' + URL_PREFIX + '/jobs/{[ job.id ]}" title="{[job.name]}">' +
                        '        {[ job.name | major_name ]}' +
                        '    </a>' +
                        '    <small>' +
                        '        {[ job.name | minor_name ]}' +
                        '    </small>' +
                        '</span>'),
        };
    });

    app.directive('dgFramework', function() {
        return {
            restrict: 'AE',
            replace: true,
            template: '<span class="badge">{[ job.framework ]}</span>',
        };
    });

    app.directive('dgHasLabels', function() {
        return {
            restrict: 'AE',
            replace: true,
            template: '<i class="glyphicon ' +
                ' {[job.has_labels ? \'glyphicon-ok\' : ' +
                '\'glyphicon-remove\']}" style="width:14px"/>',
        };
    });

    // Because jinja uses {{ and }}, tell angular to use {[ and ]}
    app.config(['$interpolateProvider', function($interpolateProvider) {
        $interpolateProvider.startSymbol('{[');
        $interpolateProvider.endSymbol(']}');
    }]);

})();
}
catch (ex) {
    console.log(ex);
}

$(document).ready(function() {

    // Ideally this should be handled with an angularjs directive, but
    // for now is a global click event.
    //  deselect when the mouse is clicked outside the table
    $(document).on('click', 'body', function(e) {
        var tag = e.target.tagName.toLowerCase();
        if (tag == 'input' || tag == 'textarea' || tag == 'button' || tag == 'a') {
            return;
        }

        var scope = angular.element(document.getElementById('all-jobs')).scope();
        scope.deselect_all();
        scope.$apply();
    });

    socket.on('task update', function(msg) {
        if (msg['update'] == 'combined_graph') {
            var scope = angular.element(document.getElementById('all-jobs')).scope();
            if (scope.set_attribute(msg['job_id'], 'sparkline', msg['data'])) {
                scope.set_attribute(msg['job_id'], 'loss', msg['data'][-1]);
                scope.$apply();
            }
        }
    });

    socket.on('job update', function(msg) {
        var scope = angular.element(document.getElementById('all-jobs')).scope();
        if (false)
            return;
        if (msg.update == 'status') {
            if (msg.status == 'Initialized' ||
                msg.status == 'Waiting' ||
                msg.status == 'Running') {
                if (scope.set_attribute(msg.job_id, 'status', msg.status) &&
                    scope.set_attribute(msg.job_id, 'status_css', msg.css))
                    scope.$apply();
            } else {
                // These should be moving from the running table to
                // the other tables, so gather all the information to
                // be displayed.
                scope.add_job(msg.job_id);
                scope.$apply();
            }
        }
        else if (msg.update == 'progress') {
            if (scope.set_attribute(msg.job_id, 'progress', msg.percentage))
                scope.$apply();
        }
        else if (msg.update == 'added') {
            scope.add_job(msg.job_id);
            scope.$apply();
        }
        else if (msg.update == 'deleted') {
            scope.remove_job(msg.job_id);
            scope.$apply();
        }
        else if (msg.update == 'attribute') {
            scope.set_attribute(msg.job_id, msg.attribute, msg.value);
            scope.$apply();
        }
    });
});
