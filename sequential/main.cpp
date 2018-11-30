#include <graphics_framework.h>
#include <glm/glm.hpp>
#include <vector>
#include <random>
#include <chrono>
#include <fstream>
#include "body.h"
#include "gif.h"
#include "main.h"

using namespace std;
using namespace glm;
using namespace graphics_framework;

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
			// Calculate force body i to body j using Newtonian gravity equation
			dvec2 f_ij = G_CONSTANT * bodies[i].mass * bodies[j].mass * dir
				/ pow(sqrt(dir.x * dir.x + dir.y * dir.y), 3);
			bodies[i].force += f_ij;
			bodies[j].force -= f_ij;
		}
	}
}

// Returns position converted to the range required for drawing
dvec2 draw_position(dvec2 p)
{
	dvec2 dp = p;
	dp -= dvec2(POS_LOWER_BOUND);
	dp /= POS_HIGHER_BOUND - POS_LOWER_BOUND;
	return dp;
}

bool setup()
{
	renderer::set_screen_dimensions(SCREEN_WIDTH, SCREEN_HEIGHT);
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
	default_random_engine rand(RANDOM_SEED); // Seed the random generator to make sure the values are consistent
	uniform_real_distribution<double> dist(POS_LOWER_BOUND, POS_HIGHER_BOUND);
	uniform_real_distribution<double> m_dist(MASS_LOWER_BOUND, MASS_HIGHER_BOUND);
	for (int i = 0; i < N_BODIES; i++)
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

	// Set up the gif writer
	GifWriter *gw = new GifWriter();
	GifBegin(gw, "simulation.gif", renderer::get_screen_width(), renderer::get_screen_height(), 1);
	// Prepare to render to frame
	renderer::setClearColour(0.0f, 0.0f, 0.0f);
	renderer::set_render_target(frame);
	renderer::bind(eff);
	// Set screen quad mvp
	glUniformMatrix4fv(eff.get_uniform_location("MVP"), 1, GL_FALSE, value_ptr(mat4(1.0f)));

	// Allocate memory to read image data into
	unique_ptr < uint8_t[]> data(new uint8_t[(renderer::get_screen_width() * renderer::get_screen_height()) * 4]);
	// Bind the frame
	glBindFramebuffer(GL_FRAMEBUFFER, frame.get_buffer());
	glPixelStorei(GL_PACK_ALIGNMENT, 1);
	glDisable(GL_DEBUG_OUTPUT);

	// Variables for measurement storage
	int iterations = 0;
	double total_time = 0;

	// Write column headers
	ofstream f("data.csv", ofstream::out);
	for (int i = 1; i <= NUMBER_OF_TESTS; i++)
		f << ", " << "Test " << i;
	f << endl;

	for (int n = 0; n < N_BODY_INCREMENTS; n++)
	{
		cout << "Starting " << NUMBER_OF_TESTS << " tests for " << N_BODIES + BODY_INCREMENT * n << " bodies." << endl;
		f << N_BODIES + BODY_INCREMENT * n << " bodies,";
		for (int test = 1; test <= NUMBER_OF_TESTS; test++)
		{
			// Simulate particles
			double time_left = DURATION;
			while (time_left > 0.0f)
			{
				// Clear frame before rendering
				glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT | GL_STENCIL_BUFFER_BIT);

				auto before = chrono::steady_clock::now();
				calculate_forces(bodies);
				auto after = chrono::steady_clock::now();
				chrono::duration<double> time_spent = after - before;
				total_time += time_spent.count();
				iterations++;

				for (int i = 0; i < bodies.size(); i++)
				{
					bodies[i].step(DELTA_TIME);
					// Only render for first test
					if (iterations == 1)
					{
						// Render each body
						glUniform2fv(eff.get_uniform_location("pos"), 1, value_ptr((vec2)draw_position(bodies[i].pos)));
						renderer::render(screen_quad);
					}
				}
				time_left -= DELTA_TIME;

				// Only write to gif for the first test
				if (iterations == 1)
				{
					// Get image data
					glReadPixels(0, 0, renderer::get_screen_width(), renderer::get_screen_height(), GL_RGBA, GL_UNSIGNED_BYTE, data.get());
					// Add frame to gif
					GifWriteFrame(gw, data.get(), renderer::get_screen_width(), renderer::get_screen_height(), 1);
				}
			}
			cout << "Test " << test << " complete. Average time spent calculating forces per frame: " << total_time / (double)iterations << endl;
			f << total_time / (double)iterations << ",";
		}
		f << endl;

	}
	// Close gif file and do cleanup
	GifEnd(gw);

	f.close();
	// Unbind framebuffer
	glBindFramebuffer(GL_FRAMEBUFFER, 0);

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