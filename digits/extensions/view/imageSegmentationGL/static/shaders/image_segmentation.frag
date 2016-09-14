// Copyright (c) 2016, NVIDIA CORPORATION.  All rights reserved.

#ifdef GL_ES
precision mediump float;
#endif

varying vec2 v_uv;

uniform sampler2D u_sampler_0;
uniform sampler2D u_sampler_1;

uniform float opacity;
uniform float line_width;
uniform float mask;
uniform float ds;
uniform float dt;
uniform vec2 zoom;

void main(void) {
    vec2 st = vec2(v_uv.s, v_uv.t);
    if (0.0 <= zoom.s && zoom.s <= 1.0 &&
        0.0 <= zoom.t && zoom.t <= 1.0) {
        float radius = 100.0;
        vec2 r = (st - zoom);
        if (length(r / vec2(ds, dt)) < radius) {
            st = 0.25 * r + zoom;
        }
    }

    gl_FragColor = texture2D(u_sampler_0, vec2(st.s, st.t));
    vec4 color = texture2D(u_sampler_1, vec2(st.s, st.t));
    if (length(color.rgb) > 0.05) {
        int in_count = 0;
        int out_count = 0;
        for (int i = -2; i <= 2; i++) {
            for (int j = -2; j <= 2; j++) {
                vec4 c = texture2D(u_sampler_1, vec2(st.s + ds * float(i) * line_width * 0.25,
                                                     st.t + dt * float(j) * line_width * 0.25));
                if (length(color - c) < 0.05)
                    in_count = in_count + 1;
                else
                    out_count = out_count + 1;
            }
        }
        if (in_count > 0 && out_count > 0)
            gl_FragColor = color;
        gl_FragColor = gl_FragColor * (1.0 - opacity) + color * opacity;
    } else {
        gl_FragColor = gl_FragColor * (1.0 - mask);
    }
    gl_FragColor.a = 1.0;
}
