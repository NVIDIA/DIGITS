var UnitHelpers = {
  defaultStyles: function(h,w){
    h = _.isUndefined(h) ? 65 : h;
    w = _.isUndefined(w) ? 65 : w;
    return {margin: "0px",height:h+"px", width:w+"px", position: "relative"}
  },
  mouseOverStyles: function(h,w){
    return {
      height: h+25+"px",
      width: w+25+"px",
    }
  },
  mouseExitStyles: function(h,w){
    return {
      height: h+"px",
      width: w+"px",
    }
  },
  defaultAttributes: function(h,w){
    h = _.isUndefined(h) ? 65 : h;
    w = _.isUndefined(w) ? 65 : w;
    return {height:h, width:w, class: "panel panel-default"};
  },
  drawUnitImage: function(image_url, canvas, size, nonSquare){
    var nonSquare = _.isUndefined(nonSquare) ? false : nonSquare;

    // Add timestamp to prevent caching
    var time_stamp = new Date().getTime();
    var image = new Image();
     image.onload = function() {
       // Get the height to width ratio
       var hw = (nonSquare == true) ? image.height/image.width : 1;
       var h = size*hw;
       var w = size;

       // Update size of canvas accordingly:
       canvas.setAttribute("height", h);
       canvas.setAttribute("width", w);
       canvas.style.width  = w+"px";
       canvas.style.height = h+"px";

       // Draw:
       var ctx = canvas.getContext("2d");
       ctx.clearRect(0, 0, w, h);
       ctx.drawImage(this, 0, 0, image.width, image.height, 0, 0, w, h);
     }
     image.src = image_url+"&time_stamp="+time_stamp;
  },
  drawImage: function(image_url,ctx,w,h){
    // Add timestamp to prevent caching
    var time_stamp = new Date().getTime();
    var image = new Image();
     image.onload = function() {
       // Get the height to width ratio
       ctx.drawImage(this, 0, 0, image.width, image.height, 0, 0, w, h);
     }
     image.src = image_url+"&time_stamp="+time_stamp;
  },
  drawUnit: function(output_data,canvas, w,h){
    // Update size of canvas accordingly:
    canvas.setAttribute("height", h);
    canvas.setAttribute("width", w);
    canvas.style.width  = w+"px";
    canvas.style.height = h+"px";

    // Draw:
    var ctx = canvas.getContext("2d");
    ctx.clearRect(0, 0, w, h);

    var grid_dim = output_data[0].length;
    var pixel_h = pixel_w = h/grid_dim;

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
  }

};
