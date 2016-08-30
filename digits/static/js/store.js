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
                  console.log($scope.groups)
              }, function errorCallback(response) {
                  $scope.local_error = response.statusText;
                  $scope.groups = null;
                  $scope.groups_length = 0;
          });
      };
      $scope.get_model_list = get_model_list;
      get_model_list(0);
  });
  app.config(['$interpolateProvider', function($interpolateProvider) {
      $interpolateProvider.startSymbol('{[');
      $interpolateProvider.endSymbol(']}');
  }]);

})(window.angular);
