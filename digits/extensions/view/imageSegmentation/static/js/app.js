// Copyright (c) 2016, NVIDIA CORPORATION.  All rights reserved.

// Angularjs app, visualization_app
var app = angular.module('visualization_app', ['ngStorage']);

// Controller to handle global display attributes
app.controller('display_controller',
               ['$scope', '$rootScope', '$localStorage',
                function($scope, $rootScope, $localStorage) {
                    $rootScope.storage = $localStorage.$default({
                        opacity: .3,
                        mask: 0.0,
                    });
                    $scope.fill_style = {'opacity': $localStorage.opacity};
                    $scope.mask_style = {'opacity': $localStorage.mask};
                    // Broadcast to child scopes that the opacity has changed.
                    $scope.$watch(function() {
                        return $localStorage.opacity;
                    }, function() {
                        $scope.fill_style = {'opacity': $localStorage.opacity};
                    });
                    // Broadcast to child scopes that the mask has changed.
                    $scope.$watch(function() {
                        return $localStorage.mask;
                    }, function() {
                        $scope.mask_style = {'opacity': $localStorage.mask};
                    });
                }]);

// Because jinja uses {{ and }}, tell angular to use {[ and ]}
app.config(['$interpolateProvider', function($interpolateProvider) {
    $interpolateProvider.startSymbol('{[');
    $interpolateProvider.endSymbol(']}');
}]);
