function generateTorchTree(nodes){
  nodes = JSON.parse(nodes);
  function getLeafNodes(leafNodes, obj){
    // Get the leaf nodes of a given branch
     if(obj.isLast != true){
         obj.children.forEach(function(child){getLeafNodes(leafNodes,child)});
     } else{
         leafNodes.push(obj);
     }
  }

  function getNodesAndLinks(nodes,links,obj){
    // Get the links between each node
    var source  = obj;
    if(obj.isLast != true){
      obj.children.forEach(function(child){
        links.push({source: source, target: child});

        if (_.pluck(nodes,"index").indexOf(child.index) != -1) return;
        nodes.push(child);
        getNodesAndLinks(nodes,links,child);

      });
    }
  }

  function toGraph(node){
    // Convert Tree Structure to Graph Structure
    var links = new Array();
    var nodes = new Array();
    var graph = {nodes: new Array(), links: new Array()};

    graph.nodes.push(node);
    getNodesAndLinks(graph.nodes, links, node);

    graph.links = links;

    return graph;
  }

  function isContainer(nodeType){
    return (nodeType == "nn.Sequential" || nodeType == "nn.Concat" || nodeType == "nn.Parallel" || nodeType == "nn.DepthConcat")
  }
  function isParallel(nodeType){
    return (nodeType == "nn.Concat" || nodeType == "nn.Parallel" || nodeType == "nn.DepthConcat")
  }

  function chainContents(node, nodes){
    // Make the parent of each sibling its previous sibling

    node.children = [node.contents[0]];
    node.contents[0].parents = [node];
    var prevNode = node;

    _.each(node.contents, function(child,i){

      child.parents = [prevNode];

      prevNode = child;

      if (isContainer(child.type)) {
        var exit = isParallel(child.type) ? branchContents(child,nodes) : chainContents(child,nodes);
        prevNode = child = exit;
        exit.isLast = false;
      }

      if (i != node.contents.length-1) child.children = [node.contents[i+1]];
      if (i == node.contents.length-1) child.isLast   = true;
    });

    // Add an exit node:
    var exitIndex = prevNode.index + 0.5;
    var exit = {index: exitIndex, type: "s-exit", children: [], parents: [], isLast: true};
    nodes.push(exit);
    prevNode.children = [exit];
    prevNode.isLast = false;
    exit.parents = [prevNode];

    return exit;

  }

  function branchContents(node,nodes){
    // Create branch structure, that terminates with the leaves
    // of each branch joining together

    var leafNodes = new Array();
    node.children = new Array();

    // Connect all of concats children to its parents
    var newParent = node.parents[0];
    newParent.children = [];

    _.each(node.contents, function(child,i){
      if (child.type == "nn.Sequential") chainContents(child,nodes);
      child.parents = [newParent];
      newParent.children.push(child);
    });

    getLeafNodes(leafNodes,newParent);
    var exitIndex = _.max(_.pluck(leafNodes, "index")) + 0.4;
    var exit = {index: exitIndex, type: "Concat", children: [], parents: [], isLast: true, chain: node.chain};
    _.each(leafNodes, function(leaf){
      leaf.children = [exit];
      leaf.isLast   = false;
      exit.parents.push(leaf);
    });

    return exit;
  }

  function removeNodes(nodes, type){
    _.each( _.filter(nodes, function(n){return n.type == type}), function(n){
      if (_.isUndefined(n.parents)) return;

      var newParent   = n.parents[0];
      var newChildren  = n.children;
      _.each(newChildren, function(c){newParent.children.push(c); c.parents = [newParent] });
      newParent.children = _.filter(newParent.children, function(c){ return c.type != type });
    });
  }

  function getContainerContents(node){
    return _.filter(nodes, function(n){return n.container.index == node.index})
  }

  _.each(nodes, function(node){
    node.contents = getContainerContents(node);
    // if (node.type == "nn.Sequential") chainContents(node);
  });

  chainContents(nodes[0], nodes);

  // Remove all exit blocks, and connect their children to their parents
  removeNodes(nodes, "s-exit");

  // Remove all sequence entry blocks and attatch to children:
  removeNodes(nodes, "nn.Sequential");

  graph = toGraph(nodes[0]);

}
