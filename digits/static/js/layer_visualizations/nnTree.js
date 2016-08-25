// Copyright (c) 2016, NVIDIA CORPORATION.  All rights reserved.
function generateNNTree(model_def,framework){
    var data = JSON.parse(model_def);
    window.graph = JSON.parse(model_def);
    if (framework == "torch"){return;}

    var layers = _.isUndefined(data.layer) ? data.layers : data.layer;

    var links = new Array();
    layers.forEach(function(layer,index){
        layer.index = index;

        // Flatten object:
        var keys = _.flatten(_.map(layer,_.keys));
        var vals = _.flatten(_.map(layer,_.values));
        var obj = _.pick(_.object(keys,vals), _.isFinite);
        _.extend(layer,obj);

        var targets = _.filter(layers,function(l){return _.contains(l.bottom,layer.top[0])});
        targets.forEach(function(target){
            var source = layer;
            var sourceTops = _.filter(layers, function(ll){ return ll.name == source.top[0] });
            // console.log(sourceTops);
            if (source.top[0] != source.name && sourceTops != 0) return
            links.push({source: source, target: target});
        });

    });

    window.graph = {nodes: layers, links: links};
}

function loadNNTree(selector){

    // size of the diagram
    var viewerWidth = 10000000;
    var viewerHeight = 10000;
    var COLORS = ["#72bee3","#f47a81","#71cdbd","#f89532","#c98fc4","#729acb","#a4de5c","#dfc000","#ed8cc3","#a084ff", "#7CD598", "#82C2FF", "#E7E84F", "#5173BD", "#DE8277","#DD3F4E","#FEE469","#CFC98D"];
    var types  = _.uniq(_.pluck(graph.nodes, "type"));
    var svg = d3.select(selector).html('').append("svg")
        .attr("width", viewerWidth)
        .attr("height", viewerHeight);

    // Create a new directed graph
    window.g = new dagreD3.graphlib.Graph().setGraph({
      rankdir: "TB",
      ranksep: "100"
    });

    // Automatically label each of the nodes
    graph.nodes.forEach(function(n,i){
      var hasName = !_.isUndefined(n.name);

      g.setNode(n.index, {
        name: hasName ? n.name : n.chain,
        label: (hasName ? n.name+"\n" : "")+n.type,
        style: "fill: "+ COLORS[types.indexOf(n.type)]
      })
    });

    // states.forEach(function(state) { g.setNode(state, { label: "BLAH" }); });
    // Set up the edges
    graph.links.forEach(function(e){
      g.setEdge(e.source.index, e.target.index,{
        label: "",
        style: "fill: none; stroke: "+COLORS[types.indexOf(e.target.type)]
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
