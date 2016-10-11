// Copyright (c) 2016, NVIDIA CORPORATION.  All rights reserved.

// Angularjs app, visualization_app
var app = angular.module('visualization_app', ['ngStorage']);

// Controller to handle global display attributes
app.controller('display_controller',
               ['$scope', '$rootScope', '$localStorage',
                function($scope, $rootScope, $localStorage) {
                    $rootScope.storage = $localStorage.$default({
                        opacity: 30,
                        mask: 0,
                        line_width: 3,
                    });
                    // Broadcast to child scopes that the opacity has changed.
                    $scope.$watch(function(){
                        return $localStorage.opacity;
                    }, function(new_opacity, old_opacity){
                        GLRenderer.opacity = new_opacity / 100.0;
                        $rootScope.$broadcast('draw_settings_changed');
                    });
                    // Broadcast to child scopes that the mask has changed.
                    $scope.$watch(function(){
                        return $localStorage.mask;
                    }, function(new_mask, old_mask){
                        GLRenderer.mask = new_mask / 100.0;
                        $rootScope.$broadcast('draw_settings_changed');
                    });
                    // Broadcast to child scopes that the line_width has changed.
                    $scope.$watch(function(){
                        return $localStorage.line_width;
                    }, function(new_line_width, old_line_width){
                        GLRenderer.line_width = new_line_width;
                        $rootScope.$broadcast('draw_settings_changed');
                    });
                    $scope.$on('draw_settings_changed', function(event) {
                        GLRenderManager.instance().settings_changed();
                    });
                }]);

// Because jinja uses {{ and }}, tell angular to use {[ and ]}
app.config(['$interpolateProvider', function($interpolateProvider) {
    $interpolateProvider.startSymbol('{[');
    $interpolateProvider.endSymbol(']}');
}]);
