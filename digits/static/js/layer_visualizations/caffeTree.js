function loadTree(selector,height){
  var height = _.isUndefined(height) ? 400 : height;
  var COLORS = ["#72bee3","#f47a81","#71cdbd","#f89532","#c98fc4","#729acb","#a4de5c","#dfc000","#ed8cc3","#a084ff", "#7CD598", "#82C2FF"];
  var types  =_.uniq(_.map(window.layers,"type"));

  // var window.treeData = data[0];
  // Calculate total nodes, max label length
  var totalNodes = 0;
  var maxLabelLength = 0;
  // variables for drag/drop
  var selectedNode = null;
  var draggingNode = null;
  // panning variables
  var panSpeed = 200;
  var panBoundary = 20; // Within 20px from edges will pan when dragging.
  // Misc. variables
  var i = 0;
  var duration = 750;
  var root;

  // size of the diagram
  var viewerWidth = 10000000;
  var viewerHeight = 10000;

  var tree = d3.layout.tree()
      .size([viewerHeight, viewerWidth]);

  // define a d3 diagonal projection for use by the node paths later on.
  var diagonal = d3.svg.diagonal()
      .projection(function(d) {
          return [d.y, d.x];
      });

  // A recursive helper function for performing some setup by walking through all nodes

  function isDown (parentName){
    return _.isUndefined(parentName) || parentName.indexOf("_down") > -1
  }

  function visit(parent, visitFn, childrenFn) {
      if (!parent) return;

      visitFn(parent);

      var children = childrenFn(parent);
      if (children) {
          var count = children.length;
          for (var i = 0; i < count; i++) {
              visit(children[i], visitFn, childrenFn);
          }
      }
  }

  // Call visit function to establish maxLabelLength
  visit(window.treeData, function(d) {
      totalNodes++;
      maxLabelLength = Math.max(d.name.length, maxLabelLength);

  }, function(d) {
      return d.children && d.children.length > 0 ? d.children : null;
  });


  // sort the tree according to the node names

  function sortTree() {
      tree.sort(function(a, b) {
          return b.name.toLowerCase() < a.name.toLowerCase() ? 1 : -1;
      });
  }
  // Sort the tree initially incase the JSON isn't in a sorted order.
  sortTree();

  // TODO: Pan function, can be better implemented.

  function pan(domNode, direction) {
      var speed = panSpeed;
      if (panTimer) {
          clearTimeout(panTimer);
          translateCoords = d3.transform(svgGroup.attr("transform"));
          if (direction == 'left' || direction == 'right') {
              translateX = direction == 'left' ? translateCoords.translate[0] + speed : translateCoords.translate[0] - speed;
              translateY = translateCoords.translate[1];
          } else if (direction == 'up' || direction == 'down') {
              translateX = translateCoords.translate[0];
              translateY = direction == 'up' ? translateCoords.translate[1] + speed : translateCoords.translate[1] - speed;
          }
          scaleX = translateCoords.scale[0];
          scaleY = translateCoords.scale[1];
          scale = zoomListener.scale();
          svgGroup.transition().attr("transform", "translate(" + translateX + "," + translateY + ")scale(" + scale + ")");
          d3.select(domNode).select('g.node').attr("transform", "translate(" + translateX + "," + translateY + ")");
          zoomListener.scale(zoomListener.scale());
          zoomListener.translate([translateX, translateY]);
          panTimer = setTimeout(function() {
              pan(domNode, speed, direction);
          }, 50);
      }
  }

  // Define the zoom function for the zoomable tree

  function zoom() {
      svgGroup.attr("transform", "translate(" + d3.event.translate + ")scale(" + d3.event.scale + ")");
  }


  // define the zoomListener which calls the zoom function on the "zoom" event constrained within the scaleExtents
  var zoomListener = d3.behavior.zoom().scaleExtent([0.1, 3]).on("zoom", zoom);


  // define the baseSvg, attaching a class for styling and the zoomListener
  var baseSvg = d3.select(selector).html('').append("svg")
      .attr("width", viewerWidth)
      .attr("height", viewerHeight)
      .attr("class", "overlay")
      .call(zoomListener);


  // Helper functions for collapsing and expanding nodes.

  function collapse(d) {
      if (d.children) {
          d._children = d.children;
          d._children.forEach(collapse);
          d.children = null;
      }
  }

  function expand(d) {
      if (d._children) {
          d.children = d._children;
          d.children.forEach(expand);
          d._children = null;
      }
  }


  // Function to center node when clicked/dropped so node doesn't get lost when collapsing/moving with large amount of children.
  function centerNode(source) {
      scale = zoomListener.scale();

      x = -source.y0;
      y = -source.x0;
      y = y * scale+height;
      d3.select('g').transition()
          .duration(duration)
          .attr("transform", "translate(" + x + "," + y + ")scale(" + scale + ")");
      zoomListener.scale(scale);
      zoomListener.translate([x, y]);
  }

  // Toggle children function
  function toggleChildren(d) {
      if (d.children) {
          d._children = d.children;
          d.children = null;
      } else if (d._children) {
          d.children = d._children;
          d._children = null;
      }
      return d;
  }

  // Toggle children on click.

  function layerClicked(d){
    var layer;
    if (d.name.indexOf("_down") > -1){
      layer = d.parent;
    }else {
      layer = d;
    }
    // Emit and event which includes the target svg object, and layer info
    var event = document.createEvent('Event');
    event.initEvent('LayerClicked', true, true);
    event.layer  = _.extend(layer, {label: layer.name });
    event.svgTarget = d3.event.target;
    document.dispatchEvent(event);
  }

  function nodeClicked(d) {
      if (d3.event.defaultPrevented ) return; // click suppressed
      if (d.name.indexOf("_down") == -1) return;
      d = toggleChildren(d);
      update(d);
  }

  var defs = d3.select("svg").append('svg:defs');

  var markerAttributes = {"refX":"8","refY":"3","markerWidth":"10","markerHeight":"10","fill":"black"};
  var arrowPath = "M0,0 L0,6 L9,3 z";

  defs.append('svg:linearGradient')
    .attr('gradientUnits', 'userSpaceOnUse')
    .attr('x1', 0).attr('y1', 0).attr('x2', 20).attr('y2', 0)
    .attr('id', 'master').call(
        function(gradient) {
          gradient.append('svg:stop').attr('offset', '0%').attr('style', 'stop-color:rgb(0,0,0);stop-opacity:0');
          gradient.append('svg:stop').attr('offset', '50%').attr('style', 'stop-color:rgb(0,0,0);stop-opacity:1');
          gradient.append('svg:stop').attr('offset', '100%').attr('style', 'stop-color:rgb(0,0,0);stop-opacity:0');
        });

  defs.selectAll('.gradient').data([0])
    .enter().append('svg:linearGradient')
    .attr('id', 'gradient')
    .attr('class', 'gradient')
    .attr('xlink:href', '#master')
    .attr('gradientTransform', function(d) { return 'translate(0.4)scale(8)'; });

  defs
    .append("marker")
      .attr(markerAttributes)
      .attr("id","arrow_end")
      .attr("orient","auto")
      .append("path","g")
        .attr("d",arrowPath);

  defs
    .append("marker")
      .attr(markerAttributes)
      .attr("id","arrow_start")
      .attr("orient","180deg")
      .append("path","g")
        .attr("d",arrowPath);

  function update(source) {
      // Compute the new height, function counts total children of root node and sets tree height accordingly.
      // This prevents the layout looking squashed when new nodes are made visible or looking sparse when nodes are removed
      // This makes the layout more consistent.
      var levelWidth  = [1];
      var boxWidth    = 155;
      var lineSpacing = 70;

      var childCount = function(level, n) {
          if (n.children && n.children.length > 0) {
              if (levelWidth.length <= level + 1) levelWidth.push(0);
              levelWidth[level + 1] += n.children.length;
              n.children.forEach(function(d) {
                  childCount(level + 1, d);
              });
          }
      };

      childCount(0, root);
      var newHeight = d3.max(levelWidth) * 25; // 25 pixels per line
      tree = tree.size([newHeight, viewerWidth]);
      tree.nodeSize([45,1]);
      // Compute the new tree layout.
      var nodes = tree.nodes(root).reverse(),
          links = tree.links(nodes);

      // Set widths between levels based on maxLabelLength.
      nodes.forEach(function(d) {
        // Alternate node distance depending if its between layers or not
        var nBoxes = Math.floor(0.5*(d.depth+1));
        var nLines = Math.ceil(0.5*(d.depth-1));

        d.y = nBoxes*boxWidth + nLines*lineSpacing;
      });

      // Update the nodes…
      node = svgGroup.selectAll("g.node").data(nodes, function(d) {return d.id || (d.id = ++i);});

      // Enter any new nodes at the parent's previous position.
      var nodeEnter = node.enter().append("g")
        .attr("class", "node")
        .attr("id", function(d){return "node-"+d.name})
        .attr("transform", function(d) {
            return "translate(" + source.y0 + "," + source.x0 + ")";
        });


      nodeEnter.append("rect")
        .attr("width", function(d){
          return isDown(d.parent.name) == false ? boxWidth : 0;
        })
        .attr({height: 40, y: -20, x: -boxWidth, stroke:"white", "stroke-width": 2})
        .style("fill",function(d){return COLORS[_.indexOf(types,d.type)]})
        .on('click', layerClicked);

      nodeEnter.append("circle")
        .attr({r: 6, stroke: "white", "stroke-width": 2})
        .attr("cx",function(d){return isDown(d.parent.name) == true ? 5: -5})
        .attr("opacity", function(d){
          return isDown(d.parent.name) == true ? 0.4: 1
          })
        .style("fill", function(d){
          return d._children ? COLORS[6] : COLORS[1];
        })
        .on("mouseover", function(d) {
          if (isDown(d.parent.name)) {return}
          d3.select(this).style("fill", "yellow");
          })
        .on("mouseout", function(d) {
          d3.select(this).style("fill", d._children ? COLORS[6] : COLORS[1]);
        })
        .on("click",nodeClicked);


      var textStyle = {"text-overflow":"clip", "overflow":"hidden", "fill-opacity": 0 , "font-size": "12px", "font-family": "sans-serif"};
      var textAttr  = {class: "nodeText", "text-anchor": "middle", x: boxWidth/2};

      nodeEnter.append("text")
        .attr(textAttr)
        .style(textStyle)
        .attr("dy","15px")
        .attr('fill', function(d, i) { return 'url(#gradient)'; })
        .text(function(d) {
            return isDown(d.parent.name) == false ? "" : d.name;
        })
        .on('click', layerClicked);


        nodeEnter.append("text")
          .attr(textAttr)
          .style(textStyle)
          .attr("dy", "-5px")
          .text(function(d) {
            return isDown(d.parent.name) == false ? "" : d.type;
          })
          .on('click', layerClicked);

      // Transition nodes to their new position.
      var nodeUpdate = node.transition()
        .duration(duration)
        .attr("transform", function(d) {
            return "translate(" + d.y + "," + d.x + ")";
        });

      // Fade the text in
      nodeUpdate.selectAll("text").style("fill-opacity", 1);

      // Transition exiting nodes to the parent's new position.
      var nodeExit = node.exit().transition()
        .duration(duration)
        .attr("transform", function(d) {
            return "translate(" + source.y + "," + source.x + ")";
        })
        .remove();

      nodeExit.select("rect").style("width", 0);

      nodeExit.selectAll("text").style("fill-opacity", 0);


      // Update the links…
      var link = svgGroup.selectAll("path.link")
        .data(links, function(d) {return d.target.id});


      // Enter any new links at the parent's previous position.
      link.enter().append("path", "g")
          .attr("class", "link")
          .style({fill: "none"})
          .style("stroke-width",function(d){
            return d.target.multi ? "2px" : "1.3px"
          })
          .style("stroke", function(d){
            return isDown(d.source.parent.name) ? "rgba(0,0,0,0)" : COLORS[_.indexOf(types,d.target.type)]
          })
          .style("marker-end",function(d){return isDown(d.source.parent.name) ? "" : "url(#arrow_end)"})
          .style("marker-start",function(d){
            if (d.target.multi == false) return "" ;
            return isDown(d.source.parent.name)  ? "" : "url(#arrow_start)"})
          .attr("d", function(d) {
              var o = {
                  x: source.x0,
                  y: source.y0
              };
              return diagonal({
                  source: o,
                  target: o
              });
          });

      // Transition links to their new position.
      link.transition()
          .duration(duration)
          .attr("d", diagonal);

      // Transition exiting nodes to the parent's new position.
      link.exit().transition()
          .duration(duration)
          .attr("d", function(d) {
              var o = {
                  x: source.x,
                  y: source.y
              };
              return diagonal({
                  source: o,
                  target: o
              });
          })
          .remove();

      // Stash the old positions for transition.
      nodes.forEach(function(d) {
          d.x0 = d.x;
          d.y0 = d.y;
      });

      root = window.treeData;

      svgGroup.selectAll(".couplings").remove();

      var lineFunction = d3.svg.line()
        .x(function(d) { return d.x; })
        .y(function(d) { return d.y; })
        .interpolate("basis");

      _.each(couplings,function(c){
        var parent = tree.nodes(root).filter(function(d) {return _.includes(d['tops'],c.parent) && d.multi == false && isDown(d.name) == true;})[0];
        var child  = tree.nodes(root).filter(function(d) {return d['top'] == c.child && d['name'] == c.name;})[0];
        if (_.isUndefined(child) || _.isUndefined(parent)){return;}
        svgGroup.append("path", "g")
        .attr("class", "couplings")
          .attr("d", function() {
            // Randomly draw connection above or below the network
            var distance = (child.x0 - parent.x0);
            var sign     = [-1,1][Math.floor(Math.random()*2)];
            var radius   = 0.16*distance*sign;


            var end = {y: child.x0,x: child.y0};
            var start = {y: parent.x0,x: parent.y0};
            var p0 = {y: parent.x0+0.5*distance,x: parent.y0 + radius};
            var p1 = {y: parent.x0+0.75*distance,x: child.y0-sign*boxWidth/2};
            return lineFunction([start,p0,p1,end]);
          })
          .style({fill: "none", "stroke-width": "1.5px", opacity: 0})
          .style("stroke",COLORS[_.indexOf(types,child.type)])
          .transition()
          .duration(1500)
          .style("opacity", 0.8);
      });

  }

  // Append a group which holds all nodes and which the zoom Listener can act upon.
  var svgGroup = baseSvg.append("g");

  // Define the root
  root = window.treeData;
  root.x0 = 0;
  root.y0 = 0;

  // Layout the tree initially and center on the root node.
  update(root);
  centerNode(root);

}
