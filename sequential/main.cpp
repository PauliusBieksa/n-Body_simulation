#include <graphics_framework.h>
#include <glm/glm.hpp>
#include <vector>
#include <random>
#include <chrono>
#include <fstream>
#include "body.h"
#include "gif.h"

using namespace std;
using namespace glm;
using namespace graphics_framework;

const double duration = 5.0f; // Duration for the simulation
const double dt = 1.0f / 50.0f;
const uint32_t n_bodies = 50;
const double G = 4.302e-3; // pc / M * (km/s)(km/s)
//const double G = 6.674e-11; // N / (kg * kg) * m * m

vector<Body> bodies;

geometry screen_quad;
effect eff;
target_camera cam;
frame_buffer frame;


// Calculates and updates forces for all bodies
void calculate_forces(vector<Body> &bodies)
{
	for (int i = 0; i < bodies.size() - 1; i++)
	{
		bodies[i].force = dvec2(0.0f);
		for (int j = i + 1; j < bodies.size(); j++)
		{
			dvec2 dir = bodies[j].pos - bodies[i].pos;
			if (dir.x == 0 && dir.y == 0)
				continue;
			dvec2 f_ij = G * bodies[i].mass * bodies[j].mass * dir
				/ pow(sqrt(dir.x * dir.x + dir.y * dir.y), 3);
			bodies[i].force += f_ij;
			bodies[j].force -= f_ij;
		}
	}
}


//// Saves the framebuffer
//void save(const std::string &filename, const bool linear, frame_buffer frame)
//{
//	// Allocate memory to read image data into
//	std::unique_ptr<unsigned char[]> data(new unsigned char[(renderer::get_screen_width() * renderer::get_screen_height())]);
//	// Bind the frame
//	glBindFramebuffer(GL_FRAMEBUFFER, frame.get_buffer());
//	glPixelStorei(GL_PACK_ALIGNMENT, 1);
//
//	if (linear) {
//		// brings data in range of (min>0) - (max<1.0) to range 1 - 254. with
//		// 0->0, 1.0->255
//		std::unique_ptr<GLfloat[]> fdata(new GLfloat[(renderer::get_screen_width() * renderer::get_screen_height())]);
//		glReadPixels(0, 0, renderer::get_screen_width(), renderer::get_screen_height(), GL_RGBA, GL_FLOAT,
//			fdata.get());
//		const double f = 1000.0;
//		const double n = 0.1;
//		float max = 0.0f; // will never be bigger than f;
//
//		for (size_t i = 0; i < (renderer::get_screen_width() * renderer::get_screen_height()); i++) {
//			if (fdata[i] != 0.0f && fdata[i] != 1.0f) {
//				fdata[i] = (2.0 * n) / (f + n - fdata[i] * (f - n));
//				max = std::max(max, fdata[i]);
//			}
//		}
//
//		for (size_t i = 0; i < (renderer::get_screen_width() * renderer::get_screen_height()); i++) {
//			if (fdata[i] == 0.0f) {
//				data[i] = 0;
//			}
//			else if (fdata[i] == 1.0f) {
//				data[i] = 255;
//			}
//			else {
//				data[i] = static_cast<unsigned char>(254.0 * (fdata[i] / max));
//			}
//		}
//	}
//	else {
//		glReadPixels(0, 0, renderer::get_screen_width(), renderer::get_screen_height(), GL_RGBA, GL_UNSIGNED_BYTE,
//			data.get());
//	}
//
//	if (CHECK_GL_ERROR) {
//		// Display error
//		std::cerr << "ERROR - Couldn't Read glReadPixels" << std::endl;
//		// Throw exception
//		throw std::runtime_error("ERROR - Couldn't Read glReadPixel");
//	}
//	stbi_flip_vertically_on_write(1);
//	const auto ret =
//		stbi_write_bmp(filename.c_str(), renderer::get_screen_width(), renderer::get_screen_height(), 1, data.get());
//	if (!ret) {
//		std::cerr << "ERROR - Can't save image" << std::endl;
//	}
//
//	// Unbind framebuffer
//	glBindFramebuffer(GL_FRAMEBUFFER, 0);
//}


bool setup()
{
	renderer::set_screen_dimensions(800, 800);
	frame = frame_buffer(renderer::get_screen_width(), renderer::get_screen_height());
	// Screen quad
	{
		vector<vec3> positions{ vec3(-1.0f, -1.0f, 0.0f), vec3(1.0f, -1.0f, 0.0f), vec3(-1.0f, 1.0f, 0.0f),	vec3(1.0f, 1.0f, 0.0f) };
		vector<vec2> tex_coords{ vec2(0.0, 0.0), vec2(1.0f, 0.0f), vec2(0.0f, 1.0f), vec2(1.0f, 1.0f) };
		screen_quad.set_type(GL_TRIANGLE_STRIP);
		screen_quad.add_buffer(positions, BUFFER_INDEXES::POSITION_BUFFER);
		screen_quad.add_buffer(tex_coords, BUFFER_INDEXES::TEXTURE_COORDS_0);
	}

	// Set up starting values for bodies
	default_random_engine rand(5); // Seed the random generator to make sure the values are consistent
	uniform_real_distribution<double> dist(0.0f, 1.0f);
	uniform_real_distribution<double> m_dist(0.2f, 0.25f);
	for (int i = 0; i < n_bodies; i++)
		bodies.push_back(Body(dvec2(dist(rand), dist(rand)), m_dist(rand)));

	// Load in shaders
	eff.add_shader("res/shaders/core.vert", GL_VERTEX_SHADER);
	eff.add_shader("res/shaders/red.frag", GL_FRAGMENT_SHADER);
	// Build effect
	eff.build();

	// Set camera properties
	cam.set_position(vec3(0.0f, 3.0f, 10.0f));
	cam.set_target(vec3(0.0f, 3.0f, 0.0f));
	cam.set_projection(quarter_pi<double>(), renderer::get_screen_aspect(), 0.1f, 1000.0f);


	GifWriter *gw = new GifWriter();
	GifBegin(gw, "simulation.gif", renderer::get_screen_width(), renderer::get_screen_height(), 1);
	renderer::setClearColour(0.0f, 0.0f, 0.0f);
	renderer::set_render_target(frame);
	// Bind effect
	renderer::bind(eff);
	// Set MVP matrix uniform
	glUniformMatrix4fv(eff.get_uniform_location("MVP"), 1, GL_FALSE, value_ptr(mat4(1.0f)));

	// Simulate particles
	double time_left = duration;
	while (time_left > 0.0f)
	{
		glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT | GL_STENCIL_BUFFER_BIT);
		calculate_forces(bodies);
		for (int i = 0; i < bodies.size(); i++)
		{
			bodies[i].step(dt);
			glUniform2fv(eff.get_uniform_location("pos"), 1, value_ptr( (vec2)bodies[i].pos));
			renderer::render(screen_quad);
		}
		time_left -= dt;
		

		////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
		// Allocate memory to read image data into
		unique_ptr < uint8_t[]> data(new uint8_t[(renderer::get_screen_width() * renderer::get_screen_height()) * 4]);
		// Bind the frame
		glBindFramebuffer(GL_FRAMEBUFFER, frame.get_buffer());
		glPixelStorei(GL_PACK_ALIGNMENT, 1);

		glReadPixels(0, 0, renderer::get_screen_width(), renderer::get_screen_height(), GL_RGBA, GL_UNSIGNED_BYTE, data.get());
		////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
		GifWriteFrame(gw, data.get(), renderer::get_screen_width(), renderer::get_screen_height(), 1);




	}
	GifEnd(gw);
	return true;
}

bool update(double delta_time)
{

	//auto current_time = chrono::system_clock::now();
	//double time_accumulator = 0.0f;
	//while (something)
	//{
	//	// Timekeeping
	//	auto new_time = chrono::system_clock::now();
	//	chrono::duration<double> frame_time = new_time - current_time;
	//	time_accumulator += frame_time.count();

	//	// Fixed delta-time loop
	//	while (time_accumulator >= dt)
	//	{
	//		calculate_forces(bodies);
	//		for (int i = 0; i < bodies.size(); i++)
	//			bodies[i].step(dt);
	//		time_accumulator -= dt;
	//	}
	//}


	//cam.update(delta_time);
	return true;
}

bool render()
{
	renderer::setClearColour(0.0f, 0.0f, 0.0f);
	// Bind effect
	renderer::bind(eff);
	// Set MVP matrix uniform
	glUniformMatrix4fv(eff.get_uniform_location("MVP"), 1, GL_FALSE, value_ptr(mat4(1.0f)));
	glUniform2fv(eff.get_uniform_location("pos"), 1, value_ptr(vec2(0.5f)));
	//renderer::render(screen_quad);
	return true;
}

int main()
{
	app application("n-body");
	application.set_load_content(setup);
	application.set_update(update);
	application.set_render(render);
	// Run application
	application.run();
	return 0;
}