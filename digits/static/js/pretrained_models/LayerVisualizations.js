var VIRDIS = ["#440154","#440256","#450457","#450559","#46075a","#46085c","#460a5d","#460b5e","#470d60","#470e61","#471063","#471164","#471365","#481467","#481668","#481769","#48186a","#481a6c","#481b6d","#481c6e","#481d6f","#481f70","#482071","#482173","#482374","#482475","#482576","#482677","#482878","#482979","#472a7a","#472c7a","#472d7b","#472e7c","#472f7d","#46307e","#46327e","#46337f","#463480","#453581","#453781","#453882","#443983","#443a83","#443b84","#433d84","#433e85","#423f85","#424086","#424186","#414287","#414487","#404588","#404688","#3f4788","#3f4889","#3e4989","#3e4a89","#3e4c8a","#3d4d8a","#3d4e8a","#3c4f8a","#3c508b","#3b518b","#3b528b","#3a538b","#3a548c","#39558c","#39568c","#38588c","#38598c","#375a8c","#375b8d","#365c8d","#365d8d","#355e8d","#355f8d","#34608d","#34618d","#33628d","#33638d","#32648e","#32658e","#31668e","#31678e","#31688e","#30698e","#306a8e","#2f6b8e","#2f6c8e","#2e6d8e","#2e6e8e","#2e6f8e","#2d708e","#2d718e","#2c718e","#2c728e","#2c738e","#2b748e","#2b758e","#2a768e","#2a778e","#2a788e","#29798e","#297a8e","#297b8e","#287c8e","#287d8e","#277e8e","#277f8e","#27808e","#26818e","#26828e","#26828e","#25838e","#25848e","#25858e","#24868e","#24878e","#23888e","#23898e","#238a8d","#228b8d","#228c8d","#228d8d","#218e8d","#218f8d","#21908d","#21918c","#20928c","#20928c","#20938c","#1f948c","#1f958b","#1f968b","#1f978b","#1f988b","#1f998a","#1f9a8a","#1e9b8a","#1e9c89","#1e9d89","#1f9e89","#1f9f88","#1fa088","#1fa188","#1fa187","#1fa287","#20a386","#20a486","#21a585","#21a685","#22a785","#22a884","#23a983","#24aa83","#25ab82","#25ac82","#26ad81","#27ad81","#28ae80","#29af7f","#2ab07f","#2cb17e","#2db27d","#2eb37c","#2fb47c","#31b57b","#32b67a","#34b679","#35b779","#37b878","#38b977","#3aba76","#3bbb75","#3dbc74","#3fbc73","#40bd72","#42be71","#44bf70","#46c06f","#48c16e","#4ac16d","#4cc26c","#4ec36b","#50c46a","#52c569","#54c568","#56c667","#58c765","#5ac864","#5cc863","#5ec962","#60ca60","#63cb5f","#65cb5e","#67cc5c","#69cd5b","#6ccd5a","#6ece58","#70cf57","#73d056","#75d054","#77d153","#7ad151","#7cd250","#7fd34e","#81d34d","#84d44b","#86d549","#89d548","#8bd646","#8ed645","#90d743","#93d741","#95d840","#98d83e","#9bd93c","#9dd93b","#a0da39","#a2da37","#a5db36","#a8db34","#aadc32","#addc30","#b0dd2f","#b2dd2d","#b5de2b","#b8de29","#bade28","#bddf26","#c0df25","#c2df23","#c5e021","#c8e020","#cae11f","#cde11d","#d0e11c","#d2e21b","#d5e21a","#d8e219","#dae319","#dde318","#dfe318","#e2e418","#e5e419","#e7e419","#eae51a","#ece51b","#efe51c","#f1e51d","#f4e61e","#f6e620","#f8e621","#fbe723","#fde725"];
var JET    = chroma.scale(["#000080", "blue", "cyan", "green", "yellow", "red", "#800000"]).colors(256);

var colormap = JET;

var LayerVisualizations = function(selector,props){
  var self = this;

  self.extend = function(props){
      props = _.isUndefined(props) ? {} : props;
      return _.extend(props, {parent: self});
  };

  self.actions  = new LayerVisualizations.Actions(self.extend(props));
  self.tree_container = d3.select(selector);

  self.carousel = null;
  self.panel    = null;
  self.overlay  = null;
  self.jobs     = null;
  self.job_id   = null;
  self.image_id = null;
  self.layer    = null;
  self.range    = null;
  self.outputs  = [];

  self.initPanel = function(props){
    var selector = self.tree_container.node();
    self.overlay = new LayerVisualizations.Overlay(selector,self.extend(props));
    self.panel   = new LayerVisualizations.Panel(selector,self.extend(props));
  };

  self.initCarousel = function(selector,props){
    self.carousel = new LayerVisualizations.Carousel(selector,self.extend(props));
  };

  self.initJobs = function(selector,props){
    self.jobs = new LayerVisualizations.Jobs(selector,self.extend(props));
    self.jobs.render();
  };

  self.dispatchInference = function() {
    if (!_.isNull(self.layer)){
      self.actions.getWeights(self.layer.name);
    }
  };

  self.update = function(){
    var h = w = 35;
    var grid_dim = self.outputs[0][0].length;
    var pixel_h = pixel_w = h/grid_dim;

    var output_style = {margin: "0px",height:h+"px", width:w+"px", position: "relative"};
    var output_attr  = {height:h, width:w, class: "panel panel-default"};

    var items = self.panel.body.selectAll("canvas").data(self.outputs);
    items.attr("class", "item").enter()
      .append("canvas")
        .attr(output_attr).style(output_style)
        .each(function(data,i){
          var ctx = this.getContext("2d");
          ctx.clearRect(0, 0, w, h);
          self.drawUnit(data,ctx,pixel_w,pixel_h);
        });

    items.exit().remove();

    // Add hover effect
    items.on("mouseover", function(data,i){
      d3.select(this).classed("canvas-hover", true);
      $(this).tooltip({title: self.range.min + i + "", placement: "bottom"});
      $(this).tooltip("show");
    });
    items.on("mouseout", function(){
      d3.select(this).classed("canvas-hover", false);
    })
  };

  self.drawOutputs = function(json){
    self.layer.stats = json.stats;
    self.panel.render();
    self.outputs.length = 0;
    self.outputs.push.apply(self.outputs, _.isUndefined(json.data) ? [] : json.data);
    if (self.outputs.length > 0 ) self.panel.body.html('');
    self.update();
    self.panel.drawNav(self.range, json.length);
  };

  self.drawUnit = function(output_data,ctx, pixel_w,pixel_h){
    _.each(output_data,function(row,iy){
        // Iterate through each row:
        _.each(row, function(pixel, ix){
          // Iterate through each column:

          var x   = ix*pixel_w;
          var y   = iy*pixel_h;
          var rgb = "";

          // if rgb array, set color to match values * 255
          if (pixel.length == 3){
            rgb = "rgb("+_.map(pixel,function(n){return Math.floor(n*255)}).join()+")";
          }else {
            // If grayscale , convert to rgb using declared color map
            // use tanh scale, instead of linear scale to match pixel color
            // for better visual appeal:
            // var c = 255*(Math.tanh(2*pixel));
            var c = 256*pixel;
            rgb = window.colormap[Math.floor(c)];
          }

          // draw pixel:
          ctx.fillStyle = rgb;
          ctx.fillRect(x,y,pixel_w,pixel_h);
          ctx.fillRect(x,y,pixel_w,pixel_h);

        });
      });
  };

  // Add listener for when a layer clicked:
  document.addEventListener("LayerClicked", function(e){
    self.layer = e.layer;
    self.range = {min: 0 , max: 500};
    self.dispatchInference();
  });

};

LayerVisualizations.Actions = function(props){
  var self   = this;
  var parent = !_.isUndefined(props.parent) ? props.parent : props;

  self.getWeights = function(layerName){

    params = $.param({
      "job_id": parent.job_id,
      "layer_name":layerName,
      "range_min": parent.range.min,
      "range_max": parent.range.max
    });

    parent.overlay.render();

    var outputs_url  = "/pretrained_models/get_weights.json?"+params;
    d3.json(outputs_url, function(error, json) {
      // console.log(json);
      parent.drawOutputs(json);
    });
  };

  self.getInference = function(layerName){

    params = $.param({
      "job_id": parent.job_id,
      "image_id":parent.image_id,
      "layer_name":layerName,
      "range_min": parent.range.min,
      "range_max": parent.range.max
    });

    parent.overlay.render();

    var outputs_url  = "/pretrained_models/get_inference.json?"+params;
    d3.json(outputs_url, function(error, json) {
      parent.drawOutputs(json);
    });
  };

  self.getOutputs = function(job_id){
    parent.overlay.render();
    var outputs_url  = "/pretrained_models/get_outputs.json?job_id="+job_id;
    d3.json(outputs_url, function(error, json) {
      parent.overlay.remove();
      parent.job_id = job_id;
      parent.jobs.load(json);
    });
  };

  self.uploadImage = function(file){
     parent.overlay.render();
     var upload_url = "/pretrained_models/upload_image.json?job_id="+parent.job_id;
     var formData = new FormData();
     // Check file type.
     if (!file.type.match('image.*')) {
       console.error("Bad File Type");
       return;
     }
     // Add the file to the request.
     formData.append('image', file, file.name);
     var xhr = new XMLHttpRequest();
     xhr.onload = function () {
        parent.overlay.remove();
        if (xhr.status === 200) {
          var json = JSON.parse(xhr.responseText);
          parent.carousel.load(json);
        } else {
          console.error("Failed to Upload File");
        }
     };
    // Send Request:
    xhr.open("POST", upload_url, true);
    xhr.send(formData);
  };

};

LayerVisualizations.Panel = function(selector,props){
  var self = this;
  var parent = !_.isUndefined(props.parent) ? props.parent : props;

  self.container = d3.select(selector);
  self.outer = null;
  self.headingCenter = null;
  self.headingRight = null;
  self.body  = null;
  self.nav   = null;

  self.drawCloseButton = function(){
    self.headingRight.append("a")
      .attr("class", "btn btn-xs btn-danger")
      .on("click",self.remove)
      .append("span").attr("class", "glyphicon glyphicon-remove");
  };

  self.updateOuputs = function(d,step){
    parent.range = {min: d*step, max: (d+1)*step};
    parent.dispatchInference();
  };

  self.drawNav = function(range,n){
    var step = range.max - range.min;
    var ul   = self.nav.html("").append("nav")
                .append("ul").attr("class", "pagination");

    var numSteps   = Math.ceil(n/step);

    ul.selectAll("li").data(_.range(0,numSteps)).enter()
      .append("li")
        .attr("class", function(d){ return (d*step == range.min) ? "active" : ""})
        .style("cursor","pointer")
        .append("a")
          .html(function(d){ return d })
          .on("click", function(d){
            self.updateOuputs(d,step);
          });
  };

  self.remove = function(){
    parent.layer = null;
    parent.overlay.remove();
  };

  self.render = function(){
    self.container.style("position","relative");

    self.outer = parent.overlay.inner.append("div").attr("class", "component-outer");
    self.outer.style("padding", "30px");

    var panel   = self.outer.append("div").attr("class", "panel panel-default");
    var heading = panel.append("div").attr("class", "panel-heading")
      .append("div").attr("class","row").style("padding","0px 10px");

    self.headingCenter = heading.append("div").attr("class", "text-center col-xs-offset-1 col-xs-10");
    self.headingCenter.append("div").html(parent.layer.label + " (" + parent.layer.stats.shape + ")");
    self.headingRight  = heading.append("div").attr("class", "col-xs-1 text-right");

    var panelBody = panel.append("div").attr("class", "panel-content");
    self.body = panelBody.append("div").attr("class", "outputs");
    self.body.append("div")
      .attr("class", "alert alert-warning")
      .style("margin","15px")
      .html("This layer contains no weights");
    self.nav = panelBody.append("div").attr("class", "panel-nav");

    self.drawCloseButton();
  };

};


LayerVisualizations.Overlay = function(selector,props){
  var self   = this;
  var props  = _.isUndefined(props) ? {} : props;
  var parent = !_.isUndefined(props.parent) ? props.parent : props;

  self.container = d3.select(selector);
  self.inner = null;

  self.render = function(){
    self.remove();
    self.container.style("position","relative");
    self.inner = self.container.append("div")
      .attr("class", "text-center component-outer loading-overlay");

    self.inner.style("background", "rgba(0,0,0,0.5)");

    var spinner = self.inner.append("span")
      .attr("class", "glyphicon glyphicon-refresh glyphicon-spin");

    spinner.style({top: "50%", "margin-top": "-20px", color: "white", "font-size": "40px"})

    return self.inner;
  };

  self.remove = function(){
    d3.selectAll(".loading-overlay").remove();
  };

};

LayerVisualizations.Carousel = function(selector,props){
  var self   = this;

  self.load   = _.noop;
  self.remove = _.noop;
  self.render = _.noop;

};

LayerVisualizations.Jobs = function(selector,props){
  var self   = this;
  var parent = !_.isUndefined(props.parent) ? props.parent : props;

  self.jobs      = props.jobs;
  self.container = d3.select(selector);
  self.tree      = null;
  self.layers    = null;

  self.update = function(){
    var items = self.container.selectAll("div").data(self.jobs);
    items.enter()
      .append("div")
        .attr("class","btn btn-xs btn-default")
        .style(LayerVisualizations.Styles.button);

    items.append("div").html(function(d){return d.name});
    items.append("div").attr("class","subtle").html(function(d){return d.id});
    items.on("click",self.dispatch);

    items.exit().remove();
  };

  self.dispatch = function(d){
    parent.carousel.remove();
    parent.actions.getOutputs(d.id);
  }

  self.load = function(json){
    if (json.framework == "caffe"){
      var d = getTreeData("text",json.model_def);
      self.layers = d.layers;
      self.tree   = d.tree;
      loadTree(parent.tree_container.node());
    } else {
      generateTorchTree(json.model_def);
      loadTorchTree(parent.tree_container.node());
    }
    parent.carousel.render(json.images);
  };

  self.render = function(d){
    self.container.style(self.styles.jobs);
    self.update();
  };

  self.styles = {
    jobs: {
      padding: "5px",
      "max-height": "200px",
      "overflow-y": "scroll",
      width: "204px",
      left: "-4px",
      position: "relative"
    }
  }

};

LayerVisualizations.Styles  = {
  button: {
    "margin-bottom": "1px",
    background: "white",
    width: "100%",
    "box-shadow": "none"
  }
}
