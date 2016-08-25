function getTreeData(type,text){

  var fileReader, lines;
  window.couplings = new Array();

  // *** Prototxt Formatting helpers ****

  // Cleans up attributes loaded from parsing a prototxt file
  function cleanAttr(attr,name){return attr.replace(name,'').replace(/["" '']/g,'');}


  // Get attributes from an string blob loaded from a prototxt file
  function getAttr(data,name){
    var attr = _.filter(data,function(d){return d.indexOf(name) > -1});
    if (attr.length > 1 && name == "bottom:"){addCouples(data,attr);}
    var d = attr[0];
    if (_.isUndefined(d)){return undefined}
    return cleanAttr(d,name);
  }

  // Get the location of all layer objects in prototxt file
  function getLayerIndeces(data){
    return _.without(_.map(data,function(d,i){
      return d.indexOf("layer {") > -1 || d.indexOf("layers {") > -1 ? i : null;
    }),null);
  }

  // Although only sort tree via one "top", keep track of all tops incase
  // they are referenced as the parent for a future non-child node
  function getTops(data){
    var tops = _.filter(data,function(d){return d.indexOf("top:") > -1});
    return _.map(tops, function(top){return cleanAttr(top,"top:")});
  }

  // If a node has multiple bottoms / inputs, add them to a separate array
  function addCouples(data,bottoms){
    var top  = getAttr(data,"top:");
    var name = getAttr(data,"name:");
    _.each(bottoms, function(b){
      var bottom = cleanAttr(b,"bottom:");
      couplings.push({parent: bottom, child: top, name: name});
    });
  }

  // *** Read Prototxt File ****
  if (type == 'file'){
    fileReader = new XMLHttpRequest();
    fileReader.open("GET", text, false);

    fileReader.onreadystatechange = function (){
      if(fileReader.readyState === 4){
        if(fileReader.status === 200 || fileReader.readyState == 4){
          // Get the lines from the protobuf file
          lines = fileReader.responseText.split('\n');
        }
      }
    }
    fileReader.send(null);
  }else {
    lines = text.split('\n');
  }


  // Get the index of each layer in the prototxt file
  var layerIndices = getLayerIndeces(lines);
  // Convert into an arrays of string layer definitions
  var unformattedLayerBlob = _.map(_.range(layerIndices.length),
    function(i){
      var start = layerIndices[i];
      var end   = layerIndices[i+1];
      return lines.slice(start,end);
    }
  );

  // Convert to array of Objects/Dict
  window.layers = _.map(unformattedLayerBlob, function(l,i){
    return {
      name: getAttr(l,"name:"),
      top: getAttr(l,"top:"),
      bottom: getAttr(l,"bottom:"),
      type: getAttr(l,"type:"),
      tops: getTops(l),
      def: l.join("\n")
    }
  });

  // Sometimes more then one Zeroth level (append a parent to all of these)
  // as to form into a proper tree structure

  var zeroth_level = _.filter(window.layers, function(l,i){return _.isUndefined(l.bottom)});
  var nth_layers = _.filter(window.layers,function(l,i){return !_.isUndefined(l.bottom)});

  if (zeroth_level.length > 0){
    var t = [{
      bottom: undefined,
      name: "Model",
      top: "model",
      type: "Model"
    }];

    zeroth_level[0].top = nth_layers[0].bottom;

    _.each(zeroth_level, function(l){
      l.bottom = "model";
    });

    window.layers = t.concat(zeroth_level).concat(nth_layers);
  }



  // Add references to the parent and children of each node:
  var n = layers.length - 1;
  _.each(layers,function(l,i){
    l["parent"]  = new Object();
    l["multi"]   = false;
    if (i > 0 ){
      _.every(_.range(i-1,-1,-1),function(j){
        var prevNode = layers[j];

        if (prevNode.top == l.bottom){
          if (prevNode.top != prevNode.bottom){
            l["parent"] = prevNode;
            (prevNode.children || (prevNode.children = [])).push(l);
            return false;
          }else {
            // Node is not a parent but rather feeds both directions
            // (likely a ReLU or Dropout layer)
            prevNode["multi"] = true;
            return true;
          }

        }
        return true;
      });
    }
  });

  // Above format difficult to display layers as it represents each node
  // as a single node (instead of a box with a downstream and upstream (top/bottom)
  // entry point). Thus convert into a better format:

  // Ensure that all ReuLU, Dropout layers etc are set to "multi"
  _.each(layers, function(l){
      if (l.type == "ReLU" || l.type == "Dropout"){
        l.multi = true;
      }
  });

  var expanedData = [];
  _.each(layers,function(l){
      var downStream = _.clone(l);
      var upstream   = l;
      downStream.children = l.children;
      _.each(downStream.children, function(n){n.parent = downStream});
      downStream.name = upstream.name + "_down";
      upstream.children = [downStream];
      expanedData.push([upstream,downStream]);
  });

  window.treeData = _.flatten(expanedData)[0];
  return {layers: window.layers, tree: window.treeData}
}
