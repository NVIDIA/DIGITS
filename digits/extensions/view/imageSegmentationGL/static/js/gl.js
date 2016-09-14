// Copyright (c) 2016, NVIDIA CORPORATION.  All rights reserved.

var vert_file = '/models/extension-static/image-segmentation-gl/shaders/image_segmentation.vert';
var frag_file = '/models/extension-static/image-segmentation-gl/shaders/image_segmentation.frag';

// Initialize the renderer
GLRenderManager.instance().initialize(vert_file, frag_file);

var gl = {};
function start_webgl(image, image0, image1, width, height) {
    // Create the GL data class for this image set
    gl[image.id] = new GL(image, image0, image1, width, height);
}
