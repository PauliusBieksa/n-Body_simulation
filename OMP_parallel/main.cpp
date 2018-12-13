#include <graphics_framework.h>
#include <glm/glm.hpp>
#include <vector>
#include <random>
#include <chrono>
#include <fstream>
#include "body.h"
#include "gif.h"
#include "main.h"
#include <omp.h>

using namespace std;
using namespace glm;
using namespace graphics_framework;

vector<Body> bodies;

geometry screen_quad;
effect eff;
target_camera cam;
frame_buffer frame;

const uint32_t thread_count = omp_get_max_threads();

// Calculates and updates forces for all bodies
void calculate_forces(vector<Body> &bodies, uint16_t chunk_size)
{
	for (Body &b : bodies)
		b.force = dvec2(0.0f);
#pragma omp parallel for num_threads(thread_count) default(none) private(chunk_size) shared(bodies) schedule(dynamic, chunk_size)
	for (int i = 0; i < bodies.size() - 1; i++)
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

	// Load in shaders
	eff.add_shader("res/shaders/core.vert", GL_VERTEX_SHADER);
	eff.add_shader("res/shaders/mass_shader.frag", GL_FRAGMENT_SHADER);
	// Build effect
	eff.build();

	// Set camera properties
	cam.set_position(vec3(0.0f, 3.0f, 10.0f));
	cam.set_target(vec3(0.0f, 3.0f, 0.0f));
	cam.set_projection(quarter_pi<double>(), renderer::get_screen_aspect(), 0.1f, 1000.0f);

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
	// Disable gl debug output so performance warnings don't pollute the console output
	glDisable(GL_DEBUG_OUTPUT);
	// Disable grawhics output window as it is not used by the program
	glfwHideWindow(renderer::get_window());

	for (uint16_t chunk = 1; chunk <= MAX_CHUNK_SIZE; chunk *= 8)
	{
		// Variables for measurement storage
		int iterations = 0;
		double total_time = 0;

		stringstream ss;
		ss << "OMP_data_chunk_size_" << chunk << ".csv";
		ofstream f(ss.str(), ofstream::out);
		// Write column headers
		for (int i = 1; i <= NUMBER_OF_TESTS; i++)
			f << ", " << "Test " << i;
		f << endl;
		for (int n = 0; n <= N_BODY_INCREMENTS; n++)
		{
			// Set up starting values for bodies
			default_random_engine rand(RANDOM_SEED); // Seed the random generator to make sure the values are consistent
			uniform_real_distribution<double> dist(POS_LOWER_BOUND, POS_HIGHER_BOUND);
			uniform_real_distribution<double> m_dist(MASS_LOWER_BOUND, MASS_HIGHER_BOUND);
			for (int i = 0; i < N_BODIES + n * BODY_INCREMENT; i++)
				bodies.push_back(Body(dvec2(dist(rand), dist(rand)), m_dist(rand)));

			// Set up the gif writer
			GifWriter *gw = new GifWriter();
			int n_of_bodies = N_BODIES + n * BODY_INCREMENT;
			string gif_name = "OMP_simulation_" + to_string(n_of_bodies) + ".gif";
			GifBegin(gw, gif_name.c_str(), renderer::get_screen_width(), renderer::get_screen_height(), 1);

			cout << "Starting " << NUMBER_OF_TESTS << " tests for " << N_BODIES + BODY_INCREMENT * n << " bodies, chunk size - " << chunk << endl;
			f << N_BODIES + BODY_INCREMENT * n << " bodies,";
			for (int test = 1; test <= NUMBER_OF_TESTS; test++)
			{
				iterations = 0;
				total_time = 0.0;
				// Simulate particles
				double time_left = DURATION;
				while (time_left > 0.0f)
				{
					// Clear frame before rendering
					glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT | GL_STENCIL_BUFFER_BIT);

					auto before = chrono::steady_clock::now();
					calculate_forces(bodies, chunk);
					auto after = chrono::steady_clock::now();
					chrono::duration<double> time_spent = after - before;
					total_time += time_spent.count();
					iterations++;

					for (int i = 0; i < bodies.size(); i++)
					{
						bodies[i].step(DELTA_TIME);
						// Only render for first test
						if (test == 1)
						{
							// Render each body
							glUniform2fv(eff.get_uniform_location("pos"), 1, value_ptr((vec2)draw_position(bodies[i].pos)));
							glUniform1d(eff.get_uniform_location("normalised_mass"), ((bodies[i].mass - MASS_LOWER_BOUND) / (MASS_HIGHER_BOUND - MASS_LOWER_BOUND)));
							renderer::render(screen_quad);
						}
					}
					time_left -= DELTA_TIME;

					// Only write to gif for the first test
					if (test == 1)
					{
						// Get image data
						glReadPixels(0, 0, renderer::get_screen_width(), renderer::get_screen_height(), GL_RGBA, GL_UNSIGNED_BYTE, data.get());
						// Add frame to gif
						GifWriteFrame(gw, data.get(), renderer::get_screen_width(), renderer::get_screen_height(), 1);
					}
				}
				cout << "Test " << test << " complete. Average time spent calculating forces per frame: " << total_time / (double)iterations << endl;
				f << total_time / (double)iterations << ",";

				if (test == 1)
				{
					// Saves the mass and final positions of the particles in a file
					ofstream f_final_pos("positions_" + to_string(n_of_bodies) + ".csv", ofstream::out);
					f_final_pos << "x, y, mass" << endl;
					for (int i = 0; i < bodies.size(); i++)
						f_final_pos << bodies[i].pos.x << ", " << bodies[i].pos.y << ", " << bodies[i].mass << endl;
					f_final_pos.close();

					// Close gif file and do cleanup
					GifEnd(gw);
				}
			}
			f << endl;

			bodies.clear();
		}
		f.close();
	}

	// Unbind framebuffer
	glBindFramebuffer(GL_FRAMEBUFFER, 0);
	renderer::shutdown();
	return true;
}

bool update(double delta_time)
{
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