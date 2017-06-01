// Copyright (c) 2016-2017, NVIDIA CORPORATION.  All rights reserved.

// Angularjs app, visualization_app
var app = angular.module('visualization_app', ['ngStorage']);

// Controller to handle global display attributes
app.controller('display_controller',
               ['$scope', '$rootScope', '$localStorage',
                function($scope, $rootScope, $localStorage) {
                    $rootScope.storage = $localStorage.$default({
                        gamma: 1.0,
                        opacity: 0.3,
                        mask: 0.0,
                        threshold: 0,
                        line_width: 3,
                        colormap: false,
                    });
                    $scope.is_binary = false;
                    $scope.build_segment = function() {
                        var c = new Array(256);
                        var a = new Array(256);
                        // The segmentation boundary distance 128
                        var threshold = $scope.is_binary ? Number($localStorage.threshold) + 128 : 128;
                        for (var i = 0; i < 256; i++) {
                            c[i] = (i < threshold ? 0.0 : 1.0);
                            a[i] = (i == 0 ? 0 :
                                i < threshold ? $localStorage.mask :
                                    i < threshold + Number($localStorage.line_width) ? 1 :
                                        $localStorage.opacity);
                        }
                        var color = c.join(',');
                        var alpha = a.join(',');
                        // For some reason, the tableValues can not reference an angular variable, so set it here.
                        if ($scope.is_binary) {
                            document.getElementById('binary-segment-r').setAttribute('tableValues', color);
                            document.getElementById('binary-segment-g').setAttribute('tableValues', color);
                            document.getElementById('binary-segment-b').setAttribute('tableValues', color);
                            document.getElementById('binary-segment-a').setAttribute('tableValues', alpha);
                        } else {
                            document.getElementById('segment-a').setAttribute('tableValues', alpha);
                        }
                    };
                    $scope.build_segment();
                    $scope.set_binary = function(v) {
                        if ($scope.is_binary == v) return;
                        $scope.is_binary = v;
                        $scope.build_segment();
                    };
                    $scope.$watch(function() {
                        return [$localStorage.opacity, $localStorage.mask,
                            $localStorage.line_width, $localStorage.threshold].join(',');
                    }, function() {
                        $scope.build_segment();
                    });
                    $scope.defaults = function() {
                        $localStorage.gamma = 1.0;
                        $localStorage.opacity = 0.3;
                        $localStorage.mask = 0.0;
                        $localStorage.threshold = 0;
                        $localStorage.line_width = 3;
                        $localStorage.colormap = false;
                    };
                }]);

// Because jinja uses {{ and }}, tell angular to use {[ and ]}
app.config(['$interpolateProvider', function($interpolateProvider) {
    $interpolateProvider.startSymbol('{[');
    $interpolateProvider.endSymbol(']}');
}]);
