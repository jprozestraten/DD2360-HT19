#include <stdio.h>
#include <sys/time.h>

#define EPSILON 1.0e-6

struct Particle
{
	float3 position;
	float3 velocity;
};


__host__ __device__ uint gen_random(uint a, uint b, uint c = 10, uint seed = 10)
{
	//printf("RANDOM NUMBER GENERATED : %d \n",(seed*a+b) % c);
	return (seed*a+b) % c;
	//return 2;
}

__device__ void update_position(Particle* part_array_gpu, float dt, uint i)
{
	part_array_gpu[i].position.x += part_array_gpu[i].velocity.x * dt;
	part_array_gpu[i].position.y += part_array_gpu[i].velocity.y * dt;
	part_array_gpu[i].position.z += part_array_gpu[i].velocity.z * dt;
}


__device__ void update_velocity(Particle* part_array_gpu, float dt, uint i, uint j)
{
	part_array_gpu[i].velocity.x += (float)(gen_random(i, j)) * dt;
	part_array_gpu[i].velocity.y += (float)(gen_random(i, j)) * dt;
	part_array_gpu[i].velocity.z += (float)(gen_random(i, j)) * dt;
} 

__global__ void updateKernel(Particle* part_array_gpu, float dt, uint j)
{
	const int i = blockIdx.x*blockDim.x + threadIdx.x;
	//printf("GPU : %.4f , je suis le thread %d \n", testgpu, i);
	update_velocity(part_array_gpu, dt, i, j);
	update_position(part_array_gpu, dt, i);
	//printf("random_num gpu : %.4f \n", (float)(gen_random(i, j)));
	float testgpu = part_array_gpu[i].position.x;
}

__host__ void updateCPU(Particle* part_array_cpu, float dt, uint j, uint NPARTICLES)
{
	float testcpu = 1.0;
	for (int i = 0; i < NPARTICLES; i++)
	{
		if(i < NPARTICLES)
		{
			testcpu = part_array_cpu[i].position.x;
			//printf("CPU : %.4f \n", testcpu);
			//printf("random_num cpu : %.d \n", gen_random(i, j));
			part_array_cpu[i].velocity.x += (float)(gen_random(i, j)) * dt;
			part_array_cpu[i].velocity.y += (float)(gen_random(i, j)) * dt;
			part_array_cpu[i].velocity.z += (float)(gen_random(i, j)) * dt;

			part_array_cpu[i].position.x += part_array_cpu[i].velocity.x * dt;
			part_array_cpu[i].position.y += part_array_cpu[i].velocity.y * dt;
			part_array_cpu[i].position.z += part_array_cpu[i].velocity.z * dt;
		}
	}
}

__host__ bool compare_float(float a, float b)
{
	return (fabs(a-b) < EPSILON);
}

__host__ bool compare_float3(float3 a, float3 b)
{
	return (compare_float(a.x, b.x) && compare_float(a.y, b.y) && compare_float(a.z, b.z));
}


__host__ bool compareParticle(Particle particle1, Particle particle2)
{
	bool result = true;
	result &= compare_float3(particle1.position, particle2.position);
	result &= compare_float3(particle1.velocity, particle2.velocity);
	return result;
}


double cpuSecond() 
{
   struct timeval tp;
   gettimeofday(&tp,NULL);
   return ((double)tp.tv_sec + (double)tp.tv_usec*1.e-6);
}

int main(int argc, char** argv)
{
	const float dt = 1.0;
	bool flag = 1;

	const int NITER = atoi(argv[1]);
	const int TPB = atoi(argv[2]);
	const int NPARTICLES = atoi(argv[3]);

	// Declare a pointer for an array of particles
	Particle* particles_gpu;
	Particle* particles_cpu;
	Particle* particles_res;

	//particles_cpu = new Particle[NPARTICLES];
	//particles_res = new Particle[NPARTICLES];
	cudaHostAlloc((void **) &particles_cpu, NPARTICLES*sizeof(Particle), cudaHostAllocDefault);
	cudaHostAlloc((void **) &particles_res, NPARTICLES*sizeof(Particle), cudaHostAllocDefault);

	// Allocate device memory to store the output array
	cudaMalloc(&particles_gpu, NPARTICLES*sizeof(Particle));

	for (int i = 0; i < NPARTICLES; i++)
	{
		if(i < NPARTICLES)
		{
			particles_cpu[i].velocity.x = 0;
			particles_cpu[i].velocity.y = 0;
			particles_cpu[i].velocity.z = 0;

			particles_cpu[i].position.x = 1;
			particles_cpu[i].position.y = 1;
			particles_cpu[i].position.z = 1;

			particles_res[i].velocity.x = 0;
			particles_res[i].velocity.y = 0;
			particles_res[i].velocity.z = 0;

			particles_res[i].position.x = 1;
			particles_res[i].position.y = 1;
			particles_res[i].position.z = 1;

		}
	}


	//double iStart_HtoD = cpuSecond();

	//double iElaps_HtoD = cpuSecond() - iStart_HtoD;

	double iStart_gpu = cpuSecond();
	//GPU computation
	for (int j = 0; j < NITER; j++) 
	{
		//printf("GPU iteration numéro %d : \n", j);
		cudaMemcpy(particles_gpu, particles_res, NPARTICLES*sizeof(Particle), cudaMemcpyHostToDevice);
		updateKernel<<<(NPARTICLES+TPB-1) / TPB, TPB>>>(particles_gpu, dt, j);
		cudaMemcpy(particles_res, particles_gpu, NPARTICLES*sizeof(Particle), cudaMemcpyDeviceToHost);
	}
	cudaDeviceSynchronize();
	//double iElaps_gpu = cpuSecond() - iStart_gpu;

	float testcpu = 1.0, testgpu = 1.0;

	//double iStart_cpu = cpuSecond();
	//CPU computation
	for (int j = 0; j < NITER; j++) 
	{
		updateCPU(particles_cpu, dt, j, NPARTICLES);
	}
	//double iElaps_cpu = cpuSecond() - iStart_cpu;


	for (int i = 0; i < NPARTICLES; i++) {
		if(i < NPARTICLES)
		{
			testcpu = particles_cpu[i].position.x; 
			testgpu = particles_res[i].position.x;
			if(!(compareParticle(particles_cpu[i], particles_res[i]))) {
				flag = 0;
				break;
			}
		}
	}

	/*double iStart_DtoH = cpuSecond();
	cudaMemcpy(particles_cpu, particles_gpu, NPARTICLES*sizeof(Particle), cudaMemcpyDeviceToHost);
	double iElaps_DtoH = cpuSecond() - iStart_DtoH; */

	printf("Comparing the output for each implementation… ");
	if (flag)
	{
		printf("Correct!\n");
	} else {
		printf("Incorrect\n");
	}
	//delete[] particles_cpu;
	//delete[] particles_res;
	cudaFree(particles_gpu);
	cudaFreeHost(particles_gpu);
	cudaFreeHost(particles_res); // Free the memory

	return 0;
}