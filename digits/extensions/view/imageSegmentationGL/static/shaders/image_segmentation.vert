// Copyright (c) 2016, NVIDIA CORPORATION.  All rights reserved.

attribute vec3 a_pos;
attribute vec2 a_uv;

uniform mat4 u_texture_matrix;
uniform mat4 u_modelview_matrix;
uniform mat4 u_projection_matrix;

varying vec2 v_uv;

void main(void) {
    gl_Position = u_projection_matrix * u_modelview_matrix * vec4(a_pos, 1.0);
    v_uv = (u_texture_matrix * vec4(a_uv, 0.0, 1.0)).st;
}
