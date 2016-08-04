var UnitHelpers = {
  defaultStyles: function(h,w){
    h = _.isUndefined(h) ? 65 : h;
    w = _.isUndefined(w) ? 65 : w;
    return {margin: "0px",height:h+"px", width:w+"px", position: "relative"}
  },
  defaultAttributes: function(h,w){
    h = _.isUndefined(h) ? 65 : h;
    w = _.isUndefined(w) ? 65 : w;
    return {height:h, width:w, class: "panel panel-default"};
  },
  drawImage: function(image_url,ctx,w,h){
    // Add timestamp to prevent caching
    var time_stamp = new Date().getTime();
    var image = new Image();
     image.onload = function() {
         ctx.drawImage(this, 0, 0, image.width, image.height, 0, 0, w, h);
     }
     image.src = image_url+"&time_stamp="+time_stamp;
  },
  drawUnit: function(output_data,ctx, w,h){

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
