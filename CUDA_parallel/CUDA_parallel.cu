#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <memory>
#include <iostream>
#include "CUDA_parallel.cuh"
#include "main.h"


// Bodies_input array contains position [0,1], velocity [2,3], mass [4]
// Bodies_output array contains position [0,1], velocity [2,3], mass [4]; mass is not used here
__global__ void forces_and_step(double *bodies_input, double *bodies_output, unsigned int count, double dt, uint16_t bods_per_thread)
{
	unsigned int index = ((blockIdx.x * blockDim.x) + threadIdx.x) * 5 * bods_per_thread;
	//unsigned int index = ((blockIdx.x * blockDim.x) + threadIdx.x) * 5;

	for (int b = 0; b < bods_per_thread; b++)
	{
		// If there are more threads than data discard the extra computations to stay in defined memory
		if (index < count * 5)
		{
			// Calculate force for this particle
			double fx = 0.0;
			double fy = 0.0;
			for (uint32_t i = 0; i < count * 5; i += 5)
			{
				double dir_x = bodies_input[i] - bodies_input[index];
				double dir_y = bodies_input[i + 1] - bodies_input[index + 1];
				// Make sure there is no division by zero
				if (dir_x == 0.0 && dir_y == 0.0)
					continue;
				fx += G_CONSTANT * bodies_input[i + 4] * bodies_input[index + 4] * dir_x
					/ pow(sqrt(dir_x * dir_x + dir_y * dir_y), 3.0);
				fy += G_CONSTANT * bodies_input[i + 4] * bodies_input[index + 4] * dir_y
					/ pow(sqrt(dir_x * dir_x + dir_y * dir_y), 3.0);
			}

			// Integration
			bodies_output[index + 2] = bodies_input[index + 2] + (fx / bodies_input[index + 4]) * dt;
			bodies_output[index + 3] = bodies_input[index + 3] + (fy / bodies_input[index + 4]) * dt;
			bodies_output[index] = bodies_input[index] + bodies_output[index + 2] * dt;
			bodies_output[index + 1] = bodies_input[index + 1] + bodies_output[index + 3] * dt;
			bodies_output[index + 4] = bodies_input[index + 4];
		}
		index += 5;
	}
}

Par_handler::Par_handler(int n_of_bodies)
{
	auto err = cudaSetDevice(0);
	size = n_of_bodies * sizeof(double) * 5;
	err = cudaMalloc((void**)&before, size);
	cudaMalloc((void**)&after, size);
}

Par_handler::~Par_handler()
{
	// Free device memory
	cudaFree(before);
	cudaFree(after);
}

void Par_handler::physics_step(std::vector<double> &bodies_formated, const double &dt, const uint16_t &bods_per_thread)
{
	// Copy data to device
	cudaMemcpy(before, &bodies_formated[0], size, cudaMemcpyHostToDevice);
	cudaDeviceProp prop;
	cudaGetDeviceProperties(&prop, 0);
	uint16_t max_threads = prop.maxThreadsPerBlock;
	uint32_t n_of_bodies = bodies_formated.size() / 5;
	uint16_t threads;
	uint16_t blocks;
	uint32_t kernels = (n_of_bodies / bods_per_thread) + ((n_of_bodies % bods_per_thread > 0) ? 1 : 0);
	if (kernels < max_threads)
	{
		threads = kernels;
		blocks = 1;
	}
	else
	{
		threads = max_threads;
		blocks = kernels / threads + 1;
	}
	// Run the kernel
	forces_and_step <<<blocks, threads>>>(before, after, n_of_bodies, dt, bods_per_thread);
	cudaDeviceSynchronize();
	// Copy data from device to host
	cudaMemcpy(&bodies_formated[0], after, size, cudaMemcpyDeviceToHost);
}