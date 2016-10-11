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

// This is the function that returns to the color from the two textures.
vec4 get_color(vec2 st) {
    // color image
    vec4 color0 = texture2D(u_sampler_0, vec2(st.s, st.t));
    // data image
    vec4 color1 = texture2D(u_sampler_1, vec2(st.s, st.t));

    // if not the background color
    if (length(color1.rgb) > 0.05) {

        // detect if the pixel is on the perimeter.
        int in_count = 0;
        int out_count = 0;
        for (int i = -1; i <= 1; i++) {
            for (int j = -1; j <= 1; j++) {
                vec4 c = texture2D(u_sampler_1, vec2(st.s + ds * float(i) * line_width,
                                                     st.t + dt * float(j) * line_width));
                if (length(color1 - c) < 0.05)
                    in_count = in_count + 1;
                else
                    out_count = out_count + 1;
            }
        }

        // if it's on the perimeter, draw it as a solid color,
        // otherwise draw with opacity.
        if (in_count > 0 && out_count > 0)
            color0 = color1;
        else
            color0 = color0 * (1.0 - opacity) + color1 * opacity;
    } else {
        // darken the non-segmented area with mask
        color0 = color0 * (1.0 - mask);
    }
    return color0;
}

void main(void) {
    float radius = 100.0;
    float shadow_width = 20.0;
    float shadow_intensity = 0.5;

    vec2 st = vec2(v_uv.s, v_uv.t);
    float rp = radius;
    // If within radius, then magnify by transforming the uv
    if (0.0 <= zoom.s && zoom.s <= 1.0 &&
        0.0 <= zoom.t && zoom.t <= 1.0) {
        vec2 r = (st - zoom);
        rp = length(r / vec2(ds, dt));
        if (rp < radius) {
            st = 0.25 * r + zoom;
        }
    }

    // Get the color
    gl_FragColor = get_color(st);

    // anti-alias the edge of the lens
    float width = 1.414;
    if (rp < radius && rp > radius - width) {
        vec4 color = get_color(vec2(v_uv.s, v_uv.t));
        float alpha = (radius - rp) / width;
        gl_FragColor = gl_FragColor * alpha + color * (1.0 - alpha);
    }

    gl_FragColor.a = 1.0;
}
