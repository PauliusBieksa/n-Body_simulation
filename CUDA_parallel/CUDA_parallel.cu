#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <memory>
#include "CUDA_parallel.cuh"
#include "main.h"

__device__ struct vec2
{
	double x;
	double y;

	//### Constructors ###//
	__device__ vec2()
	{
		x = 0.0;
		y = 0.0;
	}

	__device__ vec2(double d)
	{
		x = d;
		y = d;
	}
	
	__device__ vec2(double xx, double yy)
	{
		x = xx;
		y = yy;
	}
	//### End of constructors ###//

	//### Operators ###//
	__device__ vec2 operator* (double scalar)
	{
		vec2 v;
		v.x *= scalar;
		v.y *= scalar;
		return v;
	}

	__device__ vec2 operator/ (double scalar)
	{
		vec2 v;
		v.x /= scalar;
		v.y /= scalar;
		return v;
	}

	__device__ vec2 operator+= (vec2 other)
	{
		vec2 v = vec2(x, y);
		v.x += other.x;
		v.y += other.y;
		return v;
	}

	__device__ vec2 operator- (vec2 other)
	{
		vec2 v = vec2(x, y);
		v.x -= other.x;
		v.y -= other.y;
		return v;
	}
	//### End of operators ###//
};

// Bodies_input array contains position [0,1], velocity [2,3], mass [4]
// Bodies_output array contains position [0,1], velocity [2,3], mass [4]; mass is not used here
__global__ void forces_and_step(double *bodies_input, double *bodies_output, unsigned int count, double dt)
{
	unsigned int index = threadIdx.x; // this may need some calculation

	// Calculate force for this particle
	vec2 force;
	for (uint32_t i = 0; i < count; i++)
	{
		vec2 dir = vec2(bodies_input[i] - bodies_input[index], bodies_input[i + 1] - bodies_input[index + 1]);
		// Make sure there is no division by zero
		if (dir.x == 0.0 || dir.y == dir.y)
			continue;
		vec2 f = dir * G_CONSTANT * bodies_input[i + 4] * bodies_input[index + 4]
			/ pow(sqrt(dir.x * dir.x + dir.y * dir.y), 3);
		force += f;
	}

	// Do a step
	bodies_output[index + 2] = bodies_input[index + 2] + (force.x / bodies_input[index + 4]) * dt;
	bodies_output[index + 3] = bodies_input[index + 3] + (force.x / bodies_input[index + 4]) * dt;
	bodies_output[index] = bodies_input[index] + bodies_output[2] * dt;
	bodies_output[index + 1] = bodies_input[index + 1] + bodies_output[3] * dt;
}

Par_handler::Par_handler(int n_of_bodies)
{
	// Set up device momory
	cudaMalloc((double**)before, n_of_bodies);
	cudaMalloc((double**)after, n_of_bodies);
}

Par_handler::~Par_handler()
{
	// Free device memory
	cudaFree(before);
	cudaFree(after);
}

void Par_handler::physics_step(std::vector<double> &bodies_formated, const double &dt)
{
	// Set up the input array
	//std::vector<double> formated_data;
	auto size = sizeof(double) * bodies_formated.size();
	//for (Body b : bodies)
	//{
	//	formated_data.push_back(b.pos.x);
	//	formated_data.push_back(b.pos.y);
	//	formated_data.push_back(b.vel.x);
	//	formated_data.push_back(b.vel.y);
	//	formated_data.push_back(b.mass);
	//}

	// Copy data to device
	cudaMemcpy(before, &bodies_formated[0], size, cudaMemcpyHostToDevice);
	// Run the kernel
	forces_and_step <<<0, bodies_formated.size() >>> (before, after, bodies_formated.size(), dt);
	// Copy data from device to host
	cudaMemcpy(&bodies_formated[0], after, size, cudaMemcpyDeviceToHost);

	// Put the data in the bodies vector
	//for (int i = 0; i < bodies.size(); i++)
	//{
	//	bodies[i].pos.x = formated_data[i];
	//	bodies[i].pos.y = formated_data[i + 1];
	//	bodies[i].vel.x = formated_data[i + 2];
	//	bodies[i].vel.y = formated_data[i + 3];
	//	bodies[i].mass = formated_data[i + 4];
	//}
}