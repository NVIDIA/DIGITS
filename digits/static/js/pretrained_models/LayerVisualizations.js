// Copyright (c) 2016, NVIDIA CORPORATION.  All rights reserved.
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

    self.panel    = null;
    self.overlay  = null;
    self.jobs     = null;
    self.job_id   = null;
    self.image_id = null;
    self.layer    = null;
    self.range    = null;
    self.active_tab = null;
    self.outputs  = [];
    self.items    = [];

    // Initialization
    self.initPanel = function(props){
        // Responsible for Output Units of Inference Tasks
        var selector = self.tree_container.node();
        self.overlay = new LayerVisualizations.Overlay(selector,self.extend(props));
        self.panel   = new LayerVisualizations.Panel(selector,self.extend(props));
    };

    self.initJobs = function(selector,props){
        // Responsible for changing, and updating current job
        self.jobs = new LayerVisualizations.Jobs(selector,self.extend(props));
        self.jobs.render();
    };

    self.initTasks = function(selector,props){
        // Responsible for Inference Tasks, and Displaying Status
        self.tasks = new LayerVisualizations.Tasks(selector,self.extend(props));
        self.tasks.render();
    };


    // Dispatchers:
    self.dispatchInference = function() {
      if (_.isNull(self.layer)) return

      if (_.isNull(self.active_tab) || self.active_tab == "weights")
        self.actions.getWeights(self.layer.name);

      if (self.active_tab == "max-activations")
        self.actions.getMaxActivations(self.layer.name);

    };

    // Update:
    self.updateItem = function(item,layer,unit){
      if (self.layer.name != layer) return;
      if (self.active_tab != "max-activations") return;

      var size = 65;

      var params = $.param({
        "job_id": self.job_id,
        "layer_name":layer,
        "unit": unit,
      });

      var image_url = "/pretrained_models/max_activation?"+params;
      UnitHelpers.drawUnitImage(image_url,item,size,false);
    };

    self.update = function(){

      var items = self.items =
        self.panel.body.selectAll("canvas").data(self.outputs);

      items.attr("class", "item panel panel-default").enter()
        .append("canvas")
          .attr("class", "panel panel-default")
          .attr(UnitHelpers.defaultAttributes())
          .attr("data-layer", self.layer.name)
          .style(UnitHelpers.defaultStyles())
          .on("click", self.tasks.dispatchUnitClick)
          .each(function(data,i){
            if (_.isBoolean(data)){
              var unit = self.range.min + i;
              self.updateItem(this,self.layer.name,unit);
            }else {
              var hw = self.active_tab == "max-activations" ? 65 : 35;
              UnitHelpers.drawUnit(data,this,hw,hw);
            }
          });

      items.exit().remove();

      // Add hover effect
      items.on("mouseover", function(data,i){
        d3.select(this)
          .classed("canvas-hover", true)
          .style(UnitHelpers.mouseOverStyles(this.height,this.width));
        $(this).tooltip({title: self.range.min + i + "", placement: "bottom"});
        $(this).tooltip("show");
      });

      items.on("mouseout", function(){
        d3.select(this)
          .classed("canvas-hover", false)
          .style(UnitHelpers.mouseExitStyles(this.height,this.width));
      });

    };

    // Draw:
    self.updateTab = function(layerName,json,tabName){

        self.layer = {name: layerName};
        self.layer.stats = json.stats;
        self.active_tab = tabName;
        self.panel.render();
        self.outputs.length = 0;
        self.outputs.push.apply(self.outputs, _.isUndefined(json.data) ? [] : json.data);
        if (self.outputs.length > 0 ) self.panel.body.html('');
        self.update();
        self.tasks.render();
        self.panel.drawNav(self.range, json.length);
    };

    self.drawMaxActivations = function(layerName, json){
        self.updateTab(layerName,json,"max-activations");
    };

    self.drawWeights = function(layerName,json){
        self.updateTab(layerName,json,"weights");
    };

    self.initRange = function(){
        self.range = {min: 0 , max: 2000};
    };

    // Events:
    self.layerClicked = function(e){
        self.initRange();
        self.layer = e.layer;
      if (!_.include(self.jobs.layers_with_outputs,self.layer.name)){
          e.svgTarget.style.cursor = "not-allowed";
          return;
      }
        self.range = {min: 0 , max: 500};
        self.dispatchInference();
        self.tasks.render(e.layer.name);
    };

    self.maxActivationsFailed = function(msg){
          self.tasks.signalError(msg);
    };

    self.maxActivationsUpdated = function(msg){
      self.tasks.updateProgress(msg);
      if (msg.data.unit > self.outputs.length){
          self.initRange();
          self.dispatchInference();
          self.tasks.render(msg.data.layer);
      }else{
          var unit = msg.data.unit-1;
          var index = _.isNull(self.range) ? unit : unit-self.range.min;
          self.outputs[unit] = true;
          self.updateItem(self.items[0][index],msg.data.layer,unit);
      }
    };

    // Event Listeners:
    document.addEventListener("LayerClicked", self.layerClicked);
    socket.on('task update', self.maxActivationsUpdated);
    socket.on('task error', self.maxActivationsFailed);

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
            parent.drawWeights(layerName,json);
        });
    };

    self.stopMaxActivations = function(job_id){
        params = $.param({
          "job_id": parent.job_id,
          "gradient_ascent_id": job_id
        });

        var outputs_url  = "/pretrained_models/stop_max_activations.json?"+params;
        d3.json(outputs_url, function(error, json) {
          parent.tasks.maxActivationsStopped();
        });

    };

    self.removeMaxActivations = function(layerName){
        params = $.param({
          "job_id": parent.job_id,
          "layer_name":layerName
        });

        var outputs_url  = "/pretrained_models/remove_max_activations.json?"+params;
        d3.json(outputs_url, function(error, json) {
            self.getMaxActivations(layerName);
        });

    };

    self.postMaxActivations = function(layerName,units){
      params = $.param({
        "job_id": parent.job_id,
        "layer_name":layerName,
        "units": JSON.stringify(units)
      });

      var outputs_url  = "/pretrained_models/run_max_activations.json?"+params;
      d3.json(outputs_url).post(function(error,json){
          parent.tasks.job_id = json.job_id;
      });
    };

    self.getMaxActivations = function(layerName){
      params = $.param({
        "job_id": parent.job_id,
        "layer_name":layerName,
        "range_min": parent.range.min,
        "range_max": parent.range.max
      });
      parent.overlay.render();

      var outputs_url  = "/pretrained_models/get_max_activations.json?"+params;
      d3.json(outputs_url, function(error, json) {
          parent.drawMaxActivations(layerName,json);
      });
    };


    self.getJob = function(job_id){
        parent.overlay.render();
        var outputs_url  = "/pretrained_models/get_outputs.json?job_id="+job_id;
        d3.json(outputs_url, function(error, json) {
            parent.overlay.remove();
            parent.job_id = job_id;
            parent.jobs.load(json,error);
        });
    };

};

LayerVisualizations.Panel = function(selector,props){
    var self = this;
    var parent = !_.isUndefined(props.parent) ? props.parent : props;

    self.container = d3.select(selector);
    self.outer = null;
    self.headingCenter = null;
    self.headingRight = null;
    self.activeTab = null;
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

    self.drawHeading = function(panel){
        // Draw panel header:

        var heading = panel.append("div").attr("class", "panel-heading")
          .append("div").attr("class","row").style("padding","0px 10px");

        // In the center show buttons for weights, and max-activations:
        var headingStyles = {background: "white", margin: "0px 1px"};
        self.headingCenter = heading.append("div").attr("class", "text-center col-xs-offset-2 col-xs-8");

        // self.headingCenter.append("div").html(parent.layer.label + " (" + parent.layer.stats.shape + ")");
        self.headingCenter.append("span")
          .attr("class", "btn btn-default btn-xs")
          .style(headingStyles)
          .html("Weights")
          .attr("disabled", parent.active_tab == "weights" ? "" : null)
          .on("click", self.dispatchWeights);

        self.headingCenter.append("span")
          .attr("class", "btn btn-default btn-xs")
          .attr("disabled", parent.active_tab == "max-activations" ? "" : null)
          .style(headingStyles)
          .html("Max Activations")
          .on("click",self.dispatchGetMaxActivations);

        // On the right draw a close button
        self.headingRight  = heading.append("div").attr("class", "col-xs-2 text-right");

    };

    self.drawNav = function(range,n){
        // Display links at bottom of panel to change range of shown outputs
        if (_.isNull(range)) return;
        var step = range.max - range.min;
        var ul   = self.nav.html("").append("nav")
                    .append("ul").attr("class", "pagination");

        var numSteps   = Math.ceil(n/step);

        ul.selectAll("li").data(_.range(0,numSteps)).enter()
          .append("li")
            .attr("class", function(d){ return (d*step == range.min) ? "active" : ""})
            .style("cursor","pointer")
            .append("a")
              .html(function(d){ return d+1 })
              .on("click", function(d){
                self.updateOuputs(d,step);
              });

        var h = self.nav.node().getBoundingClientRect().height;

        self.nav.style("margin-top", -1*h + "px");
    };

      self.dispatchWeights = function(){
          parent.actions.getWeights(parent.layer.name);
      };

      self.dispatchGetMaxActivations = function(){
          parent.actions.getMaxActivations(parent.layer.name);
      };

      self.remove = function(){
          parent.layer = null;
          parent.overlay.remove();
          parent.tasks.render();
      };

      self.render = function(msg, status){
          var msg    = _.isUndefined(msg) ? "This layer contains no outputs" : msg;
          var status = _.isUndefined(status) ? "warning" : status;

          self.container.style("position","relative");

          self.outer = parent.overlay.inner.append("div").attr("class", "component-outer");
          self.outer.style("padding", "30px");

          var panel   = self.outer.append("div").attr("class", "panel panel-default");

          self.drawHeading(panel);

          var panelBody = panel.append("div").attr("class", "panel-content");

          self.body = panelBody.append("div").attr("class", "outputs");
          self.body.append("div")
            .attr("class", "alert alert-"+status)
            .style("margin","15px")
            .html(msg);
          self.nav = panelBody.append("div").attr("class", "panel-nav");

          self.drawCloseButton();
      };

    // Event Listeners:
    d3.select("body").on("keydown", function(){
        if (d3.event.keyCode == 27) self.remove();
    });
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

LayerVisualizations.Tasks = function(selector,props){
    // Shows Active Running Tasks, last clicked layer, and last clicked unit

    var self = this;
    var parent = self.parent =
      !_.isUndefined(props.parent) ? props.parent : props;

    self.container = d3.select(selector).append("div");
    self.progressButton = null;
    self.layer = null;
    self.unit = null;
    self.tasks = new Array();
    self.btn = null;
    self.job_id = null;
    self.isRunning = false;
    self.addBtn    = {status: "primary", text: "Get Max Activations"};
    self.abortBtn  = {status: "warning", text: "Abort Max Activations"};
    self.removeBtn = {status: "danger", text: "Delete Max Activations"};

    self.render = function(layer){
        self.container.html('');
        self.layer = _.isUndefined(layer) ? self.layer : layer;
        self.drawActivationsButton();

        self.drawTasks();
        self.container.append("br");
        if (!_.isNull(parent.layer)) self.drawLayerInfo();
        if (!_.isNull(self.unit)) self.drawUnitInfo();

    };

    self.dispatchUnitClick = function(data,index){
        self.unit = parent.range.min + index;
        self.render(this.dataset.layer);
    };

    self.dispatchStopMaxActivations = function(){
        parent.actions.stopMaxActivations(self.job_id);
    };

    self.dispatchRemoveMaxActivations = function(){
        parent.actions.removeMaxActivations(parent.layer.name);
    };

    self.dispatchRunMaxActivations = function(){
        self.isRunning = true;
        parent.actions.postMaxActivations(parent.layer.name, [-1]);
        self.tasks.push(new LayerVisualizations.Task(self,parent.layer.name));
        self.render();
    };

    self.drawTasks = function(){
        var container = self.container.append("div");
        self.tasks.forEach(function(task){
          task.render(container);
        });
    };

    self.drawLayerInfo = function(){
        self.container.append("i").text("Layer: ");
        self.container.append("b").text(self.layer);
        self.container.append("br");
    };

    self.drawActivationsButton = function(){
      if (_.isNull(parent.layer) || parent.active_tab != "max-activations")
        return;

      var hasOutputs = _.includes(parent.outputs, true);

      self.btn   = hasOutputs ? self.removeBtn : self.addBtn;
      self.btn   = !self.isRunning ? self.btn  : self.abortBtn;

      var action = hasOutputs ? self.dispatchRemoveMaxActivations : self.dispatchRunMaxActivations;

      var btn = self.container.append("a")
        .attr("class", "btn btn-"+self.btn.status)
        .attr("data-style", "expand-left")
        .style("width","100%")
        .on("click", !self.isRunning ? action : self.dispatchStopMaxActivations);

      btn.append("span").attr("class","ladda-label")
          .text(self.btn.text);
    };

    self.drawUnitInfo = function(){
        var hw = 180;

        var params = $.param({
          "job_id": parent.job_id,
          "layer_name":self.layer,
          "unit": self.unit,
        });

        var image_url = "/pretrained_models/max_activation?"+params;
        self.container.append("i").text("Unit: ");
        self.container.append("b").text(self.unit);
        self.container.append("br");

        var canvas = self.container.append("canvas");
        canvas.attr(UnitHelpers.defaultAttributes(hw,hw));
        canvas.style(UnitHelpers.defaultStyles(hw,hw));

        UnitHelpers.drawUnitImage(image_url,canvas.node(),hw,false);
    };

    self.getTask = function(layer){
        var task = _.last(self.tasks.filter(function(t){
          return t.layer == layer;
        }));

        return task;
    };

    self.updateProgress = function(msg){
        var task = self.getTask(msg.data.layer);
        if (_.isUndefined(task)){
          self.isRunning = true;
          self.job_id    = msg.data.id;
          self.tasks.push(new LayerVisualizations.Task(self,msg.data.layer));
          self.render();
          task = _.last(self.tasks);
        }
        task.progress = msg.data.progress;
        self.unit = msg.data.unit -1;
        self.render(msg.data.layer);
        task.update();
    };

    self.maxActivationsStopped = function(){
        var task = _.last(self.tasks.filter(function(t){
          return t.layer == self.layer;
        }));
        task.complete("warning");
    };

    self.signalError = function(msg){
        // Complete Task:
        var task = self.getTask(msg.data.layer);
        task.layer = task.layer+": "+msg.data.error;
        task.complete("danger");

        // Send error message to console:
        console.error(msg.data.error);
    };

    self.taskCompleted = function(){
        self.isRunning = false;
        self.render();
    };

};

LayerVisualizations.Task = function(parent,layer){
    var parent = parent;
    var self   = this;
    self.layer = layer;
    self.progress = 0;
    self.btn = null;
    self.btnObject = null;
    self.status = "default";
    self.container = null;

    self.complete = function(status){
        self.progress = 1;
        self.btn.stop();
        self.status = _.isUndefined(status) ? "success" : status;
        self.drawButton();
        parent.taskCompleted();
    };

    self.update = function(){
        self.btn.setProgress(self.progress);
        if (self.progress == 1) self.complete();
        self.btnObject.attr("disabled", null);
    };

    self.layerClicked = function(e){
        self.layer = e.layer;
        self.range = {min: 0 , max: 2000};
        self.dispatchInference();
        self.tasks.render(e.layer.name);
    };

    self.dispatchLayerClicked = function(){
        parent.parent.actions.getMaxActivations(self.layer);
    };

    self.drawButton = function(){
        var btn = self.btnObject =
          self.container.html('').append("a")
            .attr("class", "btn btn-"+self.status+" ladda-button")
            .style("margin-top","2px")
            .style("width","100%")
            .attr("data-style", "expand-left")
            .attr("data-size","small")
            .on("click",self.dispatchLayerClicked);

        btn.append("span").attr("class","ladda-label")
            .text(self.layer);

        if (self.progress == 1) return;
        self.btn = Ladda.create(btn.node());
        self.btn.start();
        self.btn.setProgress(self.progress);
        self.btnObject.attr("disabled", null);
    };

    self.render = function(container){
        self.container = container.append("div");
        self.drawButton();
    };

};

LayerVisualizations.Jobs = function(selector,props){
    var self   = this;
    var parent = !_.isUndefined(props.parent) ? props.parent : props;

    self.jobs      = props.jobs;
    self.container = d3.select(selector);
    self.layers_with_outputs = [];

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
        parent.actions.getJob(d.id);
    }


    self.load = function(json,error){
        try {
            generateNNTree(json.model_def,json.framework);

            if (json.framework == "caffe"){
              var softmax_layer = _.pluck(_.filter(graph.nodes, function(n){
                return n.type == "Softmax"}
              ), "name");
              self.layers_with_outputs = softmax_layer.concat(_.pluck(_.filter(
                graph.nodes, function(n){return _.has(n, "num_output")
              }),"name"));
            } else {
              self.layers_with_outputs = json.layers_with_outputs;
            }
            loadNNTree(parent.tree_container.node());
        }
        catch(err) {
            var msg = JSON.stringify(json);
            if (!_.isNull(error) && !_.isUndefined(error)){
              msg = JSON.parse(error.response).error.message;
            }
            parent.tree_container.append("div")
              .attr("class", "alert alert-danger")
              .style({height:"100%"}).html("<b>Failed to load pretrained model.</b> " + msg);
            console.error(err)
        }
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
