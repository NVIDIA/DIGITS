var COLORS, DEFAULTOPTIONS
var chartData = new Object();
var charts    = new Array();
var Range     = new Object();

COLORS = ["#72bee3","#f47a81","#71cdbd","#f89532","#c98fc4","#729acb","#a4de5c","#dfc000","#ed8cc3","#a084ff"];
COLORS = COLORS.concat(COLORS).concat(COLORS).concat(COLORS);

DEFAULTAXES = {
  type: 'linear',
  gridLines : {display : false},
  ticks: {}
};

DEFAULTOPTIONS = {
  label: "",
  height: 350,
  dataset: {
    fill: false,
    pointStyle: 'circle',
    borderWidth: 2,
    pointHitRadius: 5,
    tension: 0
  },
  isBig: true,
  axesX: DEFAULTAXES,
  axesY1: DEFAULTAXES,
  axesY2: DEFAULTAXES,
  legend: {display: false}
};


function largeGraph(allData,lrData){
  // Sets up graph inside container with id: graph
  // for large_graph.html views

  var allData = formatData(allData);
  var lrData  = formatData(lrData);
  var titles  = allData.titles.concat(lrData.titles);

  window.chartData = {axes1: allData.data, axes2: lrData.data[0]};
  generateSlider("#graph");
  window.charts.push(bigChart("#graph",titles,700,"All Network Outputs"));

}

function buildChart(selector,d1,d2,height,header){
  // Builds a single collapsable chart
  window.chartData = {axes1: d1.data, axes2: d2.data[0]};
  generateSlider(selector);
  generateExpandableChart(selector,d1.titles,height,header);
}

function buildCharts(selector,d1,d2,height){
  d3.select(selector).html("");
  // Builds many charts
  buildChart(selector,d1,d2,height+50,"All Network Outputs");
  _.each(d1.titles,function(title){
    generateExpandableChart(selector,[title],height);
  });
}

function generateSlider(selector){
  // Draw a slider to control plot range
  var sliderContainer = d3.select(selector)
  .append("div")
    .style("text-align","center")
    .style("margin","0 auto")
    .append("div")
      .attr("class","slider")
      .style("width","100%");

  var mySlider = $(sliderContainer.node()).slider({
    range: true,
    min: 0,
    max: 1,
    value: 0,
    step: 0.01
  });

  mySlider.on("slide", function(e){
    Range.set(_.object(["min","max"],e.value));
  });

}

function generateExpandableChart(selector,titles,height,header){
  // Make a container to hold small and big charts
  var id = "expandableChart" + Math.floor(Math.random()*10000);
  var container = d3.select(selector)
  .append("div")
    .attr("id",id)
    .style("width","100%")

  window.charts.push(bigChart(container.node(),titles,height,header));
  window.charts.push(smallChart(container.node(),titles,header));

  var showBig   = $("#"+id+" .showBigPlot");
  var showSmall = $("#"+id+" .showSmallPlot");

  showSmall.parent().css("display","none");

  showBig.on("click", function(){
    showBig.css("display","none");
    showSmall.parent().css("display","block");
  });

  showSmall.on("click", function(){
    showBig.css("display","block");
    showSmall.parent().css("display","none");
  });

}

function updateLegend(chart,target,i){
  var dataset = chart.config.data.datasets[i];
  d3.select(target).style("opacity", dataset.hidden ? 1 : 0.2);
  dataset.hidden = dataset.hidden ? false : true;
  chart.update();
}

function bigChart(selector,titles,height,header){
  // Draw a Big Chart
  var myChart;
  var axes    = _.extend(_.clone(DEFAULTAXES),{gridLines : {display : true}});
  var header  = _.isUndefined(header) ? titles.join() : header;

  var options = getOptions({
    height: height,
    axesY1: axes,
    axesX: axes});

  // Panel (Container)
  var myPanel = d3.select(selector).append("div")
    .attr("class","panel panel-default")
    .style("padding","5px")
    .style("margin","0 auto")
    .style("display","block")
    .style("width","100%");

  // Collapse Button
  myPanel.append("btn")
    .attr("class", "btn panel-default btn-xs showSmallPlot")
    .style("float","right")
    .append("span")
      .attr("class", "glyphicon glyphicon-remove");

  myPanel.append("p")
    .attr("class","text-center")
    .style("overflow", "auto")
    .html(header);

  var myChart = drawChart(myPanel.node(),titles,options);

  myPanel.append("div")
    .attr("class","chart-legend text-center")
    .style("max-width","100%")
    .html(myChart.generateLegend())
    .select("ul")
      .style("display", "inline-block")
      .selectAll("li")
        .attr("class","legend-item")
        .attr("style","list-style: none; float:left;margin: 0px 5px;font-size:13px")
        .on("click",function(__,i){
          var target= d3.event.target;
          if (target.className != "legend-item"){target = $(target).parent()[0];}
          updateLegend(myChart,target,i);
        })
        .select("span")
          .style("padding","0px 8px")
          .style("margin","0px 5px");

  return myChart;
}

function smallChart(selector,titles,header){
  // Draw A Small Chart
  var axesX   = _.extend(_.clone(DEFAULTAXES),{display: false});
  var axesY2  = _.extend(_.clone(DEFAULTAXES),{display: false});
  var options = getOptions({height: 50, isBig: false, axesX:axesX, axesY2:axesY2});
  var header  = _.isUndefined(header) ? titles.join() : header;

  // Panel (and Button)
  var myPanel = d3.select(selector)
  .append("a")
      .attr("class","btn panel panel-default showBigPlot")
      .style("padding","5px")
      .style("margin","0 auto")
      .style("display","block")
      .style("width","100%");

  myPanel.append("p")
    .attr("class","text-center")
    .style("max-width","100%")
    .style("overflow", "hidden")
    .html(header);

  return drawChart(myPanel.node(),titles,options);
}


function drawChart(selector, titles, options){
  // Draw a new Canvas and Chart
  var canvas, ctx, datasets, scatterChart;

  canvas = d3.select(selector)
  .append("div")
    .attr("class","chart-wrapper")
      .style("height", options.height+"px")
    .append("canvas")
      .attr("class",  "myChart")
      .attr("height", options.height);

  ctx = canvas.node().getContext("2d");

  datasets = _.filter(formatDatasets(options), function(d){
    return _.contains([window.chartData.axes2.title].concat(titles),d.title);
  });

  return new Chart(ctx, {
      type: 'line',
      data: {
          datasets: datasets
      },
      options: {
          legend: options.legend,
          scales: {
              xAxes: [_.extend({position: 'bottom'},options.axesX)],
              yAxes: [_.extend({position: 'left', id: "left"},options.axesY1),
                      _.extend({position: 'right',id: "right"},options.axesY2)],
          },
          animation: false,
          maintainAspectRatio: false,
          responsive: true
      }
  });
}

function getOptions(params){
  return _.extend(_.clone(DEFAULTOPTIONS),params);
}

function formatDatasets(options){
  // Combine Chart Data with Options
  return _.map(window.chartData.axes1, function(d,i){
    return _.extend({
      label: d.label,
      title: d.title,
      data: d.data,
      borderColor: COLORS[i],
      backgroundColor: COLORS[i],
      pointBackgroundColor: (options.isBig ? COLORS[i] : "rgba(255,255,255,0)"),
      pointBorderColor: (options.isBig ? COLORS[i] : "rgba(255,255,255,0)"),
      pointRadius: (options.isBig? 1.5 : 0),
      yAxisID: "left"
    }, options.dataset)
  }).concat(_.extend({
    label: window.chartData.axes2.label,
    title: window.chartData.axes2.title,
    data:  window.chartData.axes2.data,
    borderColor: "rgba(200,200,200,"+(options.isBig ? 1 : 0) +")",
    backgroundColor: "rgba(200,200,200,"+(options.isBig ? 1 : 0) +")",
    pointBackgroundColor: "rgba(200,200,200,"+(options.isBig ? 1 : 0) +")",
    yAxisID: "right"
  }, options.dataset));
}


Range = {
  get: function(){
    // Get the range of data in the x-axis
    var range = new Object();

    // Base range off of the first dataset with more than one point
    _.every(window.chartData.axes1, function(set,i){
      var data  = set.data;
      var x = _.map(data,function(p){return p.x});
      range = {min: _.min(x), max: _.max(x)}
      return range.min == range.max
    });

    return range;
  },
  set: function(scale){
    // Update given a scale ranging from 0 to 1
    var range = Range.get();

    var scale = _.extend({min: 0, max:1},scale);
    var new_max = scale.max*(range.max-range.min) + range.min;
    var new_min = scale.min*(range.max-range.min) + range.min;

    _.each(window.charts,function(chart){
      chart.config.options.scales.xAxes[0].ticks.min = new_min;
      chart.config.options.scales.xAxes[0].ticks.max = scale.max == 1 ? undefined : new_max;
      chart.update();
    });
  },
  reset: function(){
    // Show the entire scale
    _.each(window.charts,function(chart){
      chart.config.options.scales.xAxes[0].ticks.min = undefined;
      chart.config.options.scales.xAxes[0].ticks.max = undefined;
      chart.update();
    });
  }
}


function updateCharts(chartData1,chartData2){
  // Add new data points to Charts
  function updateSet(set,newData){
    var len   = set.data.length;
    var label = set.label;
    var set2  = _.filter(newData.data, function(item){return item.label == label})[0];
    var newData = _.rest(set2.data, len);
    _.each (newData, function(p){set.data.push(p)});
  }

  _.each(window.chartData.axes1, function(set,i){updateSet(set,chartData1);});
  updateSet(window.chartData.axes2,chartData2);
  _.each(window.charts,function(chart){chart.update()});

}


function pickData(d,x){
  // Pick data based on if it is x or y
  return _.pick(d,function(v,k){return v.isX == x});
}

function formatData(x){
  // format input data to Chart.js data object format
  var t = 'true'; var f = 'false';
  var ydata = pickData(x,f);
  var xdata = pickData(x,t);

  return {
    data: _.map(ydata, function(set,key){
      var label = key;
      var title = set.title
      var data  = _.map(set.data,function(d,i){
        return {x: xdata[set.xlabel].data[i], y: d};
      });
      return {data: data, title: title, label: key};
    }),
    titles: _.uniq(_.pluck(ydata,'title'))
  }
}
