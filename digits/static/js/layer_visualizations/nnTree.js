// Copyright (c) 2016, NVIDIA CORPORATION.  All rights reserved.
function generateNNTree(model_def,framework){
    var data = JSON.parse(model_def);
    window.graph = JSON.parse(model_def);
    if (framework == "torch"){return;}
    var blobs_names = _.map(data.layer, function(l){return l.top[0]});

    var layers = _.isUndefined(data.layer) ? data.layers : data.layer;
    var links = new Array();
    var blobs_names = data.input.concat(_.uniq(_.map(layers, function(l){return l.top[0]})));
    var blobs = _.map(blobs_names, function(n,i){return {name: n, type: "blob", index: i}});
    var offset = blobs.length;


    layers.forEach(function(layer,i){
        var index = i + offset;
        layer.index = index;

        // Flatten object:
        var keys = _.flatten(_.map(layer,_.keys));
        var vals = _.flatten(_.map(layer,_.values));
        var obj = _.pick(_.object(keys,vals), _.isFinite);
        _.extend(layer,obj);

        layer.bottom.forEach(function(bottom){
          var source = _.filter(blobs, function(b){return b.name == bottom})[0];
          links.push({source: source, target: layer});
        });

        layer.top.forEach(function(top){
          var target = _.filter(blobs, function(b){return b.name == top})[0];
          links.push({source: layer, target: target});
        });


    });
    window.graph = {nodes: blobs.concat(layers) , links: links};
}

function loadNNTree(selector){

    // size of the diagram
    var viewerWidth = 10000000;
    var viewerHeight = 10000;
    var COLORS = ["#e0e0e0", "#72bee3","#f47a81","#71cdbd","#f89532","#c98fc4","#729acb","#a4de5c","#dfc000","#ed8cc3","#a084ff", "#7CD598", "#82C2FF", "#E7E84F", "#5173BD", "#DE8277","#DD3F4E","#FEE469","#CFC98D"];
    var types  = _.uniq(_.pluck(graph.nodes, "type"));
    var svg = d3.select(selector).html('').append("svg")
        .attr("width", viewerWidth)
        .attr("height", viewerHeight);

    // Create a new directed graph
    window.g = new dagreD3.graphlib.Graph().setGraph({
      rankdir: "TB",
      ranksep: 50,
      edgesep: 1,
      nodesep: 10
    });

    // Automatically label each of the nodes
    graph.nodes.forEach(function(n,i){
      var hasName = !_.isUndefined(n.name);
      var type = (n.type == "blob") ? "" : n.type;
      g.setNode(n.index, _.extend({
        name: hasName ? n.name : n.chain,
        label: (hasName ? n.name+"\n" : "")+type,
        style: "fill: "+ COLORS[types.indexOf(n.type)],
        shape: (n.type == "blob") ? "ellipse" : "rect"
      },
      (n.type == "blob") ? {width: 150} : {}));
    });

    // states.forEach(function(state) { g.setNode(state, { label: "BLAH" }); });
    // Set up the edges
    graph.links.forEach(function(e){
      var num_sources = _.isUndefined(e.target.bottom) ? 1 : e.target.bottom.length;

      g.setEdge(e.source.index, e.target.index,{
        label: "",
        style: "fill: none; stroke: "+COLORS[types.indexOf(e.target.type)],
        lineInterpolate: "basis",
        minLen: num_sources > 1 ? 2 : 1
      })
    });

    var inner = svg.append("g").attr("transform", "translate(" + 0 + "," + 0 + ")scale(" + 0.1 + ")");

    // Set up zoom support
    var zoom = d3.behavior.zoom().on("zoom", function() {
          inner.attr("transform", "translate(" + d3.event.translate + ")" +
                                      "scale(" + d3.event.scale + ")");
        });
    svg.call(zoom);
    // Create the renderer
    var render = new dagreD3.render();
    // Run the renderer. This is what draws the final graph.
    render(inner, g);
    // Center the graph
    var topNode = d3.select("#tree_container > svg > g > g > g.nodes > g:nth-child(1)");
    var container = d3.select("#tree_container");
    var containerWidth = container.node().getBoundingClientRect().width;
    var left  = parseFloat(topNode.attr("transform").match("translate(.*),")[1].slice(1));
    var width = topNode.node().getBoundingClientRect().width;
    var initX = -1*left + containerWidth/2;
    var initY = 0;
    var initScale = 1;

    inner.transition().duration(800).attr("transform", "translate(" + initX + "," + initY + ")scale(" + initScale + ")");

    zoom
      .translate([initX, initY])
      .scale();

    // Add click listeners to each layer:
    inner.selectAll("g.node")
      .on("click", function(d) {
        // Emit and event which includes the target svg object, and layer info
        var event = document.createEvent('Event');
        event.initEvent('LayerClicked', true, true);
        console.log(g.node(d).name);
        event.layer  = {name: g.node(d).name, label: g.node(d).label};
        event.svgTarget = d3.event.target;
        document.dispatchEvent(event);
    });
}
