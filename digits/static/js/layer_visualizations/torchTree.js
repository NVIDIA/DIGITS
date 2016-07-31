function loadTorchTree(selector){
  // size of the diagram
  var viewerWidth = 10000000;
  var viewerHeight = 10000;
  var COLORS = ["#72bee3","#f47a81","#71cdbd","#f89532","#c98fc4","#729acb","#a4de5c","#dfc000","#ed8cc3","#a084ff", "#7CD598", "#82C2FF"];
  var types  = _.uniq(_.pluck(graph.nodes, "type"));
  var svg = d3.select(selector).html('').append("svg")
      .attr("width", viewerWidth)
      .attr("height", viewerHeight);

  // Create a new directed graph
  var g = new dagreD3.graphlib.Graph().setGraph({rankdir: "LR"});
  // States and transitions from RFC 793

  // Automatically label each of the nodes
  graph.nodes.forEach(function(n,i){g.setNode(n.index, {label: n.type, chain: n.chain, style: "fill: "+ COLORS[types.indexOf(n.type)]})});

  // states.forEach(function(state) { g.setNode(state, { label: "BLAH" }); });
  // Set up the edges
  graph.links.forEach(function(e){
    g.setEdge(e.source.index, e.target.index,{label: "", style: "fill: none; stroke: "+COLORS[types.indexOf(e.target.type)]})
  });

  g.nodes().forEach(function(v) {
    var node = g.node(v);
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

  var initX = 50;
  var initY = 400;
  var initScale = 1.2;

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
      event.layer  = {name: g.node(d).chain, label: g.node(d).label};
      event.svgTarget = d3.event.target;
      document.dispatchEvent(event);
  });
}
