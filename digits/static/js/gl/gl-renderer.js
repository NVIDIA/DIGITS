// Copyright (c) 2016, NVIDIA CORPORATION.  All rights reserved.

"use strict";

////////////////////////////////////////////////////////////////////////////////
function GLRenderer() {
    // This will be a gl canvas
    this.canvas = document.createElement('canvas');
    this.canvas.className = "gl-canvas";

    this.attached = undefined;

    this.pos_buffer = undefined;
    this.uv_buffer = undefined;
    this.index_buffer = undefined;
    this.program = undefined;
    this.texture_count = 0;
    this._shaders = {};

    this.modelview_matrix = mat4.create();
    this.projection_matrix = mat4.create();
    this.texture_matrix = mat4.create();

    this.mouse_pos = {x: -2.0, y: -2.0};

    // This will load the shader text from files, and once loaded,
    // will call run_webgl().
    this.load_shader_files();
}

GLRenderer.SHADER_TYPE_FRAGMENT = "x-shader/x-fragment";
GLRenderer.SHADER_TYPE_VERTEX = "x-shader/x-vertex";

// single gl render canvas which gets reparented depending on focus
// class members
GLRenderer._vert_file = undefined;
GLRenderer._frag_file = undefined;
GLRenderer._all_GLs = {};

// Display settings
GLRenderer.opacity = 0.3;
GLRenderer.mask = 0.0;
GLRenderer.line_width = 5.0;

// object methods
GLRenderer.prototype = {
    getContext: function() {
        try {
            var context = this.canvas.getContext("webgl", {preserveDrawingBuffer: true});
        } catch (e) {
            console.log(e);
        }
        if (!context) {
            alert("Unable to initialize WebGL.");
        }
        return context;
    },

    resize: function(width, height) {
        var context = this.getContext();
        context.viewportWidth = this.canvas.width = width;
        context.viewportHeight = this.canvas.height = height;
    },

    textures_loaded: function() {
        return this.texture_count == 2;
    },

    disable_drawing: function() {
        this.canvas.style.display = 'none';
    },

    enable_drawing: function() {
        this.canvas.style.display = '';
    },

    textures_loaded_callback: function() {
        this.enable_drawing();
    },

    both_shaders_loaded: function() {
        return (GLRenderer._vert_file in this._shaders &&
                GLRenderer._frag_file in this._shaders);
    },

    load_shader_file: function(file, type) {
        var xhr = new XMLHttpRequest();
        var that = this;
        xhr.addEventListener("load", function(data) {
            that._shaders[file] = {script: data.target.response, type: type};
            if (that.both_shaders_loaded()) {
                that.run_webgl();
            }
        });
        xhr.open("GET", file + '?ver=' + Date.now());
        xhr.send();
    },

    load_shader_files: function() {
        this.load_shader_file(GLRenderer._vert_file, GLRenderer.SHADER_TYPE_VERTEX);
        this.load_shader_file(GLRenderer._frag_file, GLRenderer.SHADER_TYPE_FRAGMENT);
    },

    set_matrices: function() {
        var gl = this.getContext();
        gl.uniformMatrix4fv(this.program.projection_matrix, false, this.projection_matrix);
        gl.uniformMatrix4fv(this.program.modelview_matrix, false, this.modelview_matrix);
        gl.uniformMatrix4fv(this.program.texture_matrix, false, this.texture_matrix);
    },

    handle_loaded_texture: function(texture) {
        var gl = this.getContext();
        gl.bindTexture(gl.TEXTURE_2D, texture);
        gl.pixelStorei(gl.UNPACK_FLIP_Y_WEBGL, true);
        gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, gl.RGBA, gl.UNSIGNED_BYTE, texture.image);
        gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.NEAREST);
        gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.NEAREST);
        gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
        gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
        gl.bindTexture(gl.TEXTURE_2D, null);
    },

    get_shader: function(file) {
        var shader_object = this._shaders[file];
        var shader_script = shader_object.script;
        var shader_type = shader_object.type;

        var gl = this.getContext();
        var shader;
        if (shader_type == GLRenderer.SHADER_TYPE_FRAGMENT) {
            shader = gl.createShader(gl.FRAGMENT_SHADER);
        } else if (shader_type == GLRenderer.SHADER_TYPE_VERTEX) {
            shader = gl.createShader(gl.VERTEX_SHADER);
        } else {
            return null;
        }

        gl.shaderSource(shader, shader_script);
        gl.compileShader(shader);

        if (!gl.getShaderParameter(shader, gl.COMPILE_STATUS)) {
            alert(gl.getShaderInfoLog(shader));
            return null;
        }

        return shader;
    },

    init_buffers: function() {
        var gl = this.getContext();

        // Create buffers for a single unit quad from two triangles with unit texture coordinates.
        this.pos_buffer = gl.createBuffer();
        gl.bindBuffer(gl.ARRAY_BUFFER, this.pos_buffer);
        var vertices = [
            0.0, 0.0,  1.0,
            1.0, 0.0,  1.0,
            1.0, 1.0,  1.0,
            0.0, 1.0,  1.0,
        ];
        gl.bufferData(gl.ARRAY_BUFFER, new Float32Array(vertices), gl.STATIC_DRAW);
        this.pos_buffer.itemSize = 3;
        this.pos_buffer.numItems = 4;

        this.uv_buffer = gl.createBuffer();
        gl.bindBuffer(gl.ARRAY_BUFFER, this.uv_buffer);
        var uvs = [
            0.0, 0.0,
            1.0, 0.0,
            1.0, 1.0,
            0.0, 1.0,
        ];
        gl.bufferData(gl.ARRAY_BUFFER, new Float32Array(uvs), gl.STATIC_DRAW);
        this.uv_buffer.itemSize = 2;
        this.uv_buffer.numItems = 4;

        this.index_buffer = gl.createBuffer();
        gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, this.index_buffer);
        var indices = [
            0, 1, 2,
            0, 2, 3,
        ];
        gl.bufferData(gl.ELEMENT_ARRAY_BUFFER, new Uint16Array(indices), gl.STATIC_DRAW);
        this.index_buffer.itemSize = 1;
        this.index_buffer.numItems = 6;
    },

    init_program: function() {
        // create the shaders from the loaded text files
        var vertex_shader = this.get_shader(GLRenderer._vert_file);
        var fragment_shader = this.get_shader(GLRenderer._frag_file);

        // create a shader program and attach the shaders
        var gl = this.getContext();
        this.program = gl.createProgram();
        gl.attachShader(this.program, vertex_shader);
        gl.attachShader(this.program, fragment_shader);
        gl.linkProgram(this.program);

        if (!gl.getProgramParameter(this.program, gl.LINK_STATUS)) {
            alert("Could not initialize main shaders");
        }

        gl.useProgram(this.program);

        // we need vertex positions
        this.program.vertexPositionAttribute = gl.getAttribLocation(this.program, "a_pos");
        gl.enableVertexAttribArray(this.program.vertexPositionAttribute);

        // we need texture coordinates
        this.program.textureCoordAttribute = gl.getAttribLocation(this.program, "a_uv");
        gl.enableVertexAttribArray(this.program.textureCoordAttribute);

        // set up the common uniforms
        this.program.projection_matrix = gl.getUniformLocation(this.program, "u_projection_matrix");
        this.program.modelview_matrix = gl.getUniformLocation(this.program, "u_modelview_matrix");
        this.program.texture_matrix = gl.getUniformLocation(this.program, "u_texture_matrix");
        this.program.sampler_0 = gl.getUniformLocation(this.program, "u_sampler_0");
        this.program.sampler_1 = gl.getUniformLocation(this.program, "u_sampler_1");
    },

    init_shaders: function() {
        var gl = this.getContext();

        // set up the specific uniforms
        this.program.opacity = gl.getUniformLocation(this.program, "opacity");
        this.program.mask = gl.getUniformLocation(this.program, "mask");
        this.program.line_width = gl.getUniformLocation(this.program, "line_width");
        this.program.ds = gl.getUniformLocation(this.program, "ds");
        this.program.dt = gl.getUniformLocation(this.program, "dt");
        this.program.zoom = gl.getUniformLocation(this.program, "zoom");
    },

    init_textures: function() {
        var gl = this.getContext();

        this.texture_0 = gl.createTexture();
        this.texture_0.image = new Image();
        var that = this;
        this.texture_0.image.onload = function () {
            that.handle_loaded_texture(that.texture_0);
            that.texture_count++;
            if (that.textures_loaded())
                that.textures_loaded_callback();
        }

        this.texture_1 = gl.createTexture();
        this.texture_1.image = new Image();
        this.texture_1.image.onload = function () {
            that.handle_loaded_texture(that.texture_1);
            that.texture_count++;
            if (that.textures_loaded())
                that.textures_loaded_callback();
        }
    },

    bind_buffers: function() {
        var gl = this.getContext();

        gl.bindBuffer(gl.ARRAY_BUFFER, this.pos_buffer);
        gl.vertexAttribPointer(this.program.vertexPositionAttribute,
                               this.pos_buffer.itemSize, gl.FLOAT, false, 0, 0);

        gl.bindBuffer(gl.ARRAY_BUFFER, this.uv_buffer);
        gl.vertexAttribPointer(this.program.textureCoordAttribute,
                               this.uv_buffer.itemSize, gl.FLOAT, false, 0, 0);

        gl.activeTexture(gl.TEXTURE0);
        gl.bindTexture(gl.TEXTURE_2D, this.texture_0);
        gl.uniform1i(this.program.sampler_0, 0);

        gl.activeTexture(gl.TEXTURE1);
        gl.bindTexture(gl.TEXTURE_2D, this.texture_1);
        gl.uniform1i(this.program.sampler_1, 1);
    },

    draw_scene: function() {
        if (!this.textures_loaded())
            return;

        var gl = this.getContext();

        gl.viewport(0, 0, gl.viewportWidth, gl.viewportHeight);
        gl.clear(gl.COLOR_BUFFER_BIT | gl.DEPTH_BUFFER_BIT);

        // bind the common buffers and textures
        this.bind_buffers();

        gl.uniform1f(this.program.opacity, GLRenderer.opacity);
        gl.uniform1f(this.program.mask, GLRenderer.mask);
        gl.uniform1f(this.program.line_width, GLRenderer.line_width);
        gl.uniform1f(this.program.ds, 1.0 / gl.viewportWidth);
        gl.uniform1f(this.program.dt, 1.0 / gl.viewportHeight);
        gl.uniform2f(this.program.zoom, this.mouse_pos.x, this.mouse_pos.y);

        gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, this.index_buffer);

        mat4.ortho(this.projection_matrix, 0.0, 1.0, 0.0, 1.0, -1.0, 1.0);
        mat4.identity(this.modelview_matrix);
        mat4.identity(this.texture_matrix);

        this.set_matrices();
        gl.drawElements(gl.TRIANGLES, this.index_buffer.numItems, gl.UNSIGNED_SHORT, 0);
    },

    tick: function() {
        var that = this;
        requestAnimFrame(function() {
            that.tick()
        });
        this.draw_scene();
    },

    get_mouse_pos: function(event) {
        var rect = this.canvas.getBoundingClientRect();
        return {
            x: (event.clientX - rect.left) / rect.width,
            y: 1.0 - (event.clientY - rect.top) / rect.height,
        };
    },

    attach: function(id) {
        if ((this.attached != undefined && this.attached.image.id == id) ||
            !(id in GLRenderer._all_GLs)) {
            return;
        }

        this.disable_drawing();
        this.attached = GLRenderer._all_GLs[id];
        this.texture_count = 0;
        this.texture_0.image.src = this.attached.image0;
        this.texture_1.image.src = this.attached.image1;
        this.resize(this.attached.image.width, this.attached.image.height);
        this.attached.image.parentElement.appendChild(this.canvas);
    },

    run_webgl: function() {
        this.init_buffers();
        this.init_program();
        this.init_shaders();
        this.init_textures();

        var gl = this.getContext();
        gl.clearColor(0.0, 0.0, 0.0, 1.0);
        gl.enable(gl.DEPTH_TEST);
    },
};


////////////////////////////////////////////////////////////////////////////////
function GLFGRenderer() {
    GLRenderer.apply(this, arguments);
}
GLFGRenderer.prototype = inherit(GLRenderer.prototype)

GLFGRenderer.prototype.run_webgl = function(){
    GLRenderer.prototype.run_webgl.apply(this, arguments);

    var that = this;
    document.onmousemove = function(event) {
        if (event.target.nodeName.toLowerCase() == 'img') {
            that.attach(event.target.id);
            that.mouse_pos = that.get_mouse_pos(event);
        } else if (event.target.nodeName.toLowerCase() == 'canvas') {
            that.mouse_pos = that.get_mouse_pos(event);
        } else {
            that.mouse_pos = {x: -2.0, y: -2.0};
        }
    };
    this.tick();
};

////////////////////////////////////////////////////////////////////////////////
function GLBGRenderer() {
    GLRenderer.apply(this, arguments);
    this.busy_count = 0;
    this.time = Date.now();
    this._queue = [];
}
GLBGRenderer.prototype = inherit(GLRenderer.prototype)

GLBGRenderer.prototype.textures_loaded_callback = function() {
    if (this.attached != undefined) {
        // draw again without zoom
        this.mouse_pos = {x: -2.0, y: -2.0};
        this.draw_scene();

        var data = this.canvas.toDataURL();
        if (data != undefined) {
            this.attached.image.src = this.canvas.toDataURL();
            this.attached.time = this.time;
        }
    }
    this.busy_count = 0;
};

GLBGRenderer.prototype.update_queue = function() {
    // Safety net: if it's been busy this long, skip this one.
    this.busy_count %= 20;

    if (this.busy_count > 0) {
        this.busy_count++;
        return;
    }

    // If the queue is empty add a lower priority item. These would be
    // images that outof view, but which need to be updated. Only add
    // one, so that the higher priority items are dealt with quickly.
    if (this._queue.length == 0) {
        for (var id in GLRenderer._all_GLs) {
            if (GLRenderer._all_GLs[id].time != this.time) {
                this.enqueue(id);
                break;
            }
        }
    }

    if (this._queue.length == 0)
        return;

    var id = this._queue.shift();

    if (!(id in GLRenderer._all_GLs)) {
        return;
    }

    if (GLRenderer._all_GLs[id].time == this.time) {
        return;
    }

    this.busy_count = 1;
    this.attached = GLRenderer._all_GLs[id];

    this.texture_count = 0;
    this.resize(this.attached.width, this.attached.height);
    this.texture_0.image.src = this.attached.image0;
    this.texture_1.image.src = this.attached.image1;
};

GLBGRenderer.prototype.run_webgl = function(){
    GLRenderer.prototype.run_webgl.apply(this, arguments);

    var that = this;
    var intervalID = window.setInterval(function(){that.update_queue();}, 50);
};

GLBGRenderer.prototype.enqueue = function(id) {
    this._queue.push(id);
};

GLBGRenderer.prototype.update_time = function() {
    this.time = Date.now();
};

////////////////////////////////////////////////////////////////////////////////
function getViewportHeight() {
    var height = window.innerHeight; // Safari, Opera
    var mode = document.compatMode;

    if ( (mode || !$.support.boxModel) ) { // IE, Gecko
        height = (mode == 'CSS1Compat') ?
            document.documentElement.clientHeight : // Standards
            document.body.clientHeight; // Quirks
    }

    return height;
}

////////////////////////////////////////////////////////////////////////////////
function GLRenderManager() {
    this._fg = undefined;
    this._bg = undefined;
}
GLRenderManager._instance = undefined;
GLRenderManager.instance = function() {
    if (GLRenderManager._instance === undefined)
        GLRenderManager._instance = new GLRenderManager();
    return GLRenderManager._instance;
}

// class methods
GLRenderManager.prototype = {
    initialize: function(vert_file, frag_file) {
        GLRenderer._vert_file = vert_file;
        GLRenderer._frag_file = frag_file;
        // create the renderer if it doesn't exist
        this._fg = new GLFGRenderer();
        this._bg = new GLBGRenderer();
    },

    settings_changed: function() {
        if (this._bg == undefined)
            return;

        this._bg.update_time();

        for (var id in GLRenderer._all_GLs) {
            if (GLRenderer._all_GLs[id].in_view()) {
                this.enqueue(id);
            }
        }
    },

    enqueue: function(id) {
        if (this._bg == undefined)
            return;
        this._bg.enqueue(id);
    },

    on_resize: function() {
        if (this._fg != undefined && this._fg.attached != undefined) {
            this._fg.resize(this._fg.attached.image.width,
                            this._fg.attached.image.height);
        }
    },
};

////////////////////////////////////////////////////////////////////////////////
function GL(image, image0, image1, width, height) {
    this.image = image;
    this.image0 = image0;
    this.image1 = image1;
    this.width = width;
    this.height = height;
    this.time = 0;
    GLRenderer._all_GLs[image.id] = this;
}

GL.prototype.in_view = function() {
    if (this.image === undefined) return false;

    var top = 0;
    var bottom = (window.innerHeight ||
                  document.documentElement.clientHeight);
    var left = 0;
    var right = (window.innerWidth ||
                  document.documentElement.clientWidth);

    var rect = this.image.getBoundingClientRect();

    return (rect.top < bottom && rect.bottom > top &&
            rect.left < right && rect.right > left);
};

function inherit(proto) {
	function F() {}
	F.prototype = proto
	return new F
}
