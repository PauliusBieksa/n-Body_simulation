#version 440

// Model view projection matrix
uniform mat4 MVP;

// Incoming value for the position
layout(location = 0) in vec3 position;
layout(location = 10) in vec2 uv;

layout(location = 2) out vec2 uv_out;

// Main vertex shader function
void main() {
  // Calculate screen position of vertex
  gl_Position = MVP * vec4(position, 1.0);
  uv_out = uv;
}