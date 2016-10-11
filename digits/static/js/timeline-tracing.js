//# Copyright (c) 2016-2017, NVIDIA CORPORATION.  All rights reserved.
'use strict';
var TimelineTrace = function(cont_id) {
    var container = document.createElement('track-view-container');
    container.id = 'track_view_container';
    this.viewer = document.createElement('tr-ui-timeline-view');
    this.viewer.track_view_container = container;
    Polymer.dom(this.viewer).appendChild(container);
    this.viewer.id = 'trace-viewer';
    this.viewer.globalMode = false;
    this.viewer.viewTitle = 'No Trace Selected';
    Polymer.dom(document.getElementById(cont_id)).appendChild(this.viewer);
};

TimelineTrace.prototype.onResult = function(step, result) {
    var model = new tr.Model();
    var viewer = this.viewer;
    var i = new tr.importer.Import(model);
    var p = i.importTracesWithProgressDialog([result]);
    function onModelLoaded() {
        viewer.model = model;
        viewer.viewTitle = 'Trace #' + step;
    }
    function onImportFail() {
        var overlay = new tr.ui.b.Overlay();
        overlay.textContent = tr.b.normalizeException(err).message;
        overlay.title = 'Import error';
        overlay.visible = true;
    }
    p.then(onModelLoaded, onImportFail);
};
