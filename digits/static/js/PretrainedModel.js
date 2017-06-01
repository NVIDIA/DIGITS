// Copyright (c) 2016-2017, NVIDIA CORPORATION.  All rights reserved.
var PretrainedModel = function(params) {
  var props = _.extend({
    selector: '#pretrainedModelContent',
    size: 600
  },params);

  var self = this;
  var inputs = PretrainedModel.mixins;

  self.actions = new PretrainedModel.Actions(self);
  self.archive = new Object();

  self.size = props.size;
  self.selector = props.selector;

  self.header = null;
  self.container = null;
  self.innerContainer = null;
  self.frameworkSelector = null;

  self.frameworks = [
    {text: 'Caffe', value: 'caffe'},
    {text: 'Torch', value: 'torch'}
  ];

  self.resize_channels = [
    {text: 'Color', value: 3},
    {text: 'Grayscale', value: 1}
  ];

  self.resize_modes = [
    {text: 'Squash', value: 'squash'},
    {text: 'Crop', value: 'crop'},
    {text: 'Fill', value: 'fill'},
    {text: 'Half Crop, Half Fill', value: 'half_crop'}
  ];

  self.frameworkChanged = function() {
    var nextFramework = self.frameworkSelector.property('value');
    if (nextFramework == 'torch') {
      self.torchForm();
    }else {
      self.caffeForm();
    }
  };

  self.newRow = function() {
    return self.container.append('div').attr('class', 'row');
  };

  self.close = function() {
    d3.select(self.selector).html('');
    d3.select('.modalOuter').style('display', 'none');
  };

  self.render = function() {
    // Render heading:
    $('#pretrainedModelTab>a').click();
    d3.select(self.selector).html('');
    self.heading();

    self.container = d3.select(self.selector).append('div')
      .html('')
      .style({
        top: '-20px',
        position: 'relative',
        padding: '10px ' + self.size / 10 + 'px'
      });

    self.well();
    var row = self.newRow();
    inputs.field(row.append('div').attr('class', 'col-xs-6'), 'text', 'Jobname', 'job_name');

    self.frameworkSelector = inputs.select(
      row.append('div').attr('class', 'col-xs-6'), self.frameworks, 'Framework', 'framework'
    );
    row = self.newRow();
    row.style('border-radius', '5px 5px 0px 0px');
    inputs.select(
      row.append('div').attr('class', 'col-xs-6'),
        self.resize_channels, 'Image Type', 'image_type'
    );

    inputs.select(
      row.append('div').attr('class', 'col-xs-6'),
        self.resize_modes, 'Resize Mode', 'resize_mode'
    );

    row = self.newRow();
    row.style('border-radius', '0px 0px 5px 5px');
    inputs.field(row.append('div').attr('class', 'col-xs-6'), 'number', 'Width', 'width').attr('value', 256);
    inputs.field(row.append('div').attr('class', 'col-xs-6'), 'number', 'Height', 'height').attr('value', 256);

    self.frameworkSelector.on('change', self.frameworkChanged);
    self.innerContainer = self.container.append('div');
    self.caffeForm();
  };

  self.dispatchUpload = function() {
    var file = self.archive.input.node().files[0];
    // var spinner = self.archive.button.select("span").text("")
    //   .attr("class", "");
    self.well({
      text: '',
      class: 'glyphicon glyphicon-refresh glyphicon-spin'
    });
    self.actions.uploadArchive(file);
  };

  self.well = function(params) {
    var props = _.extend({
      text: 'Upload Tar or Zip Archive',
      class: '',
      state: 'primary'
    },params);

    self.container.select('.well').remove();

    var well = self.container.insert('div', ':first-child')
      .attr('class', 'well text-center')
      .style('padding', '0px');

    well.append('div').attr('class', 'btn btn-sm btn-default')
      .style('margin', '3px')
      .attr('disabled', '')
      .html('Manual Entry');

    var archive = self.archive;

    archive.button = well.append('div')
      .attr('class', 'file-upload btn btn-sm btn-' + props.state)
      .style({position: 'relative', display: 'inline-block !important'});


    archive.button.append('span').attr('class', props.class).text(props.text);

    archive.input = archive.button.append('input').attr({
      type: 'file',
      class: 'upload',
      multiple: ''
    });

    archive.input.node().onchange = self.dispatchUpload;

  };

  self.heading = function(params) {
    var props = _.extend({
      text: 'Upload Pretrained Model',
      classed: false
    },params);

    if (!_.isNull(self.header)) self.header.remove();

    var header =
    self.header = d3.select(self.selector)
      .classed('panel-danger', props.classed)
      .insert('div', ':first-child')
        .attr('class', 'panel-heading text-center');

    header.append('span').html(props.text);
    header.append('a')
      .attr('class', 'btn btn-danger btn-xs closeButton')
      .style('float', 'right')
      .on('click', self.close)
      .append('span')
        .attr('class', 'glyphicon glyphicon-remove');

    d3.select(self.selector).append('br');

  };

  self.caffeForm = function() {
    self.innerContainer.html('');

    inputs.file(self.innerContainer, 'Weights (**.caffemodel)', 'weights_file');
    inputs.file(self.innerContainer, 'Model Definition (original.prototxt)', 'model_def_file');
    inputs.file(self.innerContainer, 'Labels file: (labels.txt)', 'labels_file');

    self.innerContainer.append('button').attr({type: 'submit', class: 'btn btn-default'})
      .on('click', self.submit)
      .style('background', 'white')
      .html('Upload Model');
  };

  self.torchForm = function(e) {
    self.innerContainer.html('');

    inputs.file(self.innerContainer, 'Weights (**.t7)', 'weights_file');
    inputs.file(self.innerContainer, 'Model Definition: (model.lua)', 'model_def_file');
    inputs.file(self.innerContainer, 'Labels file: (Optional)', 'labels_file');

    self.innerContainer.append('button').attr({type: 'submit', class: 'btn btn-default'})
      .on('click', self.submit)
      .style('background', 'white')
      .html('Upload Model');
  };

};

PretrainedModel.Actions = function(props) {
  var self = this;
  var props = _.isUndefined(props) ? {} : props;
  var parent = !_.isUndefined(props.parent) ? props.parent : props;

  self.uploadArchive = function(file) {
       var upload_url = '/pretrained_models/upload_archive';
       parent.heading({classed: false, text: 'Uploading archive, one moment...'});

       var formData = new FormData();
       // Check file type.
       if (file.type.indexOf('zip') == -1) {
         parent.heading({classed: true, text: 'Bad File Type'});
         parent.well({class: '', text: 'Try Upload Again?', state: 'danger'});
         console.error('Bad File Type');
         return;
       }

       // Add the file to the request.
       formData.append('archive', file, file.name);
       var xhr = new XMLHttpRequest();
       xhr.onload = function() {
          if (xhr.status === 200) {
            var json = JSON.parse(xhr.responseText);
            $('#pretrainedModelTab>a').click();
            parent.close();
          } else {
            parent.heading({classed: true, text: 'Upload Failed'});
            parent.well({class: '', text: 'Try Upload Again?', state: 'danger'});
            console.error('Failed to Upload File');
            console.error(xhr.responseText);
            var json = JSON.parse(xhr.responseText);
            console.error(json.status);
            parent.heading({classed: true, text: json.status});
          }
       };
      // Send Request:
      xhr.open('POST', upload_url, true);
      xhr.send(formData);
    };
};

PretrainedModel.mixins = {
  select: function(obj, data, label, name) {
    var group = obj.append('div').attr('class', 'form-group');
    group.append('label')
        .attr('for', name).html(label);

    var mySelect = group.append('select').attr({
        class: 'form-control',
        name: name
      });

    mySelect.selectAll('option').data(data).enter()
      .append('option')
        .attr('value', function(d) {return d.value})
        .text(function(d) {return d.text});

    return mySelect;

  },

  file: function(obj, label, name) {
    var group = obj.append('div').attr('class', 'input-group').style({
      margin: '6px 0px',
      width: '100%'
    });

    // Show file select as a button:
    var btn = group.append('span').attr('class', 'input-group-btn')
      .append('span')
        .attr('class', 'btn btn-default btn-file')
        .style({width: '205px', background: 'whitesmoke'});
    btn.append('span').style('font-size', '12px').text(label);
    var input = btn.append('input').attr({type: 'file', name: name});

    // Draw textfield beside button:
    var textfield = group.append('input').attr({
      type: 'text',
      class: 'form-control',
      readonly: ''
    });
    // When file selected, fill text-field with the filename
    input.on('change', function() {
        var name = this.files[0].name;
        textfield.attr('value', name);
    });
    return group;
  },

  field: function(obj, type, label, name) {
    var group = obj.append('div').attr('class', 'form-group');
    group.append('label')
        .attr('for', name).html(label);
    return group.append('input')
      .attr({type: type, class: 'form-control', name: name});
  }

};
