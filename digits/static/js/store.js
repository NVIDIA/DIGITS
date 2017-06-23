// Copyright (c) 2016-2017, NVIDIA CORPORATION.  All rights reserved.

(function(angular) {
    'use strict';
    var app = angular.module('modelStore', []);
    app.controller('ModelListController', function($scope, $http) {
        var end_point = '/store/models';
        var get_model_list = function(refresh) {
            $http({
                method: 'GET',
                url: end_point,
                params: {refresh: refresh}
            }).then(function successCallback(response) {
                $scope.groups = response.data;
                $scope.local_error = null;
                $scope.groups_length = Object.keys($scope.groups).length;
            }, function errorCallback(response) {
                errorAlert(response);
                $scope.local_error = response.statusText;
                $scope.groups = null;
                $scope.groups_length = 0;
            });
        };
        $scope.get_model_list = get_model_list;
        get_model_list(0);

        $scope.download = function(id) {
            $http({
                method: 'GET',
                url: '/store/push?id=' + id,
            }).then(function successCallback(response) {
                // nothing
            }, function errorCallback(response) {
                errorAlert(response);
            });
        };

        $scope.set_attribute = function(model_id, name, value) {
            for (var key in $scope.groups) {
                var model_list = $scope.groups[key].model_list;
                for (var i in model_list) {
                    if (model_list[i].id === model_id) {
                        model_list[i][name] = value;
                        return true;
                    }
                }
            }
            return false;
        };
    });
    app.config(['$interpolateProvider', function($interpolateProvider) {
        $interpolateProvider.startSymbol('{[');
        $interpolateProvider.endSymbol(']}');
    }]);
})(window.angular);

$(document).ready(function() {
    socket.on('update', function(msg) {
        var scope = angular.element(document.getElementById('modelList')).scope();
        if (msg.update == 'progress') {
            if (scope.set_attribute(msg.model_id, 'progress', msg.progress))
                scope.$apply();
        }
    });
});
