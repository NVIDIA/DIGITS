(function() {
    "use strict";
    var app = angular.module('ModelStoreApp', ['ui.router']);
    app.config(['$stateProvider', '$urlRouterProvider', function($stateProvider, $urlRouterProvider) {
        $urlRouterProvider.otherwise('/list');
        $stateProvider
        .state('home', {
            url: '/',
            templateUrl: '/static/partials/modelList.html',
            controller: 'modelListController'
        })
        .state('modelList', {
            url: '/list',
            templateUrl: '/static/partials/modelList.html',
            controller: 'modelListController'
        })
        .state('modelPublish', {
            url: '/add',
            templateUrl: '/static/partials/modelPublish.html',
            controller: 'modelPublishController'
        })
    }]);
    app.controller('modelListController', ["$scope", "$http", function($scope, $http) {
      $scope.GetModels = function() {
        $http.get('/store').then(
          function(response) { $scope.zoo = response.data; }
        )
      }
    }]);
    app.controller('modelPublishController', function($scope, $http) {});
}());
