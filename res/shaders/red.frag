#version 440

uniform vec2 pos;

// Incoming UV
layout(location = 2) in vec2 uv;

// Outgoing colour for the shader
layout(location = 0) out vec4 out_colour;
void main()
{
	vec2 to_pos = pos - uv;
	if (dot(to_pos, to_pos) < 0.00001)
		out_colour = vec4(1.0, 0.0, 0.0, 1.0);
	else
		discard;
}