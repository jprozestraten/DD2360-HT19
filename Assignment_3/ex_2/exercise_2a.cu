#include <stdio.h>
#include <sys/time.h>

struct Particle
{
	float3 position;
	float3 velocity;
};

__host__ __device__ uint gen_random(uint a, uint b, uint c = 10, uint seed = 10)
{
	return (seed*a+b) % c;
}

__device__ float3 update_position(float3 position, float3 velocity, float dt)
{
	position.x += velocity.x * dt;
	position.y += velocity.y * dt;
	position.z += velocity.z * dt;
	return position;
}


__device__ float3 update_velocity(float3 velocity, float dt, uint i, uint j)
{
	velocity.x += gen_random(i, j) * dt;
	velocity.y += gen_random(i, j) * dt;
	velocity.z += gen_random(i, j) * dt;
	return velocity;
}

__global__ void updateKernel(Particle* part_array_gpu, float dt, uint j)
{
	const int i = blockIdx.x*blockDim.x + threadIdx.x;
	part_array_gpu[i].velocity = update_velocity(part_array_gpu[i].velocity, dt, i, j);
	part_array_gpu[i].position = update_position(part_array_gpu[i].position, part_array_gpu[i].velocity, dt);
}

__host__ void updateCPU(Particle* part_array_cpu, float dt, uint j, uint NPARTICLES)
{
	for (int i = 0; i < NPARTICLES; i++)
	{
		part_array_cpu[i].velocity.x += gen_random(i, j) * dt;
		part_array_cpu[i].velocity.y += gen_random(i, j) * dt;
		part_array_cpu[i].velocity.z += gen_random(i, j) * dt;

		part_array_cpu[i].position.x += part_array_cpu[i].velocity.x * dt;
		part_array_cpu[i].position.y += part_array_cpu[i].velocity.y * dt;
		part_array_cpu[i].position.z += part_array_cpu[i].velocity.z * dt;
	}
}

__host__ bool compareParticle(Particle particle1, Particle particle2)
{
	bool result = true;

	result &= particle1.position.x == particle2.position.x;
	result &= particle1.position.y == particle2.position.y;
	result &= particle1.position.z == particle2.position.z;

	result &= particle1.velocity.x == particle2.velocity.x;
	result &= particle1.velocity.y == particle2.velocity.y;
	result &= particle1.velocity.z == particle2.velocity.z;

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

	particles_cpu = new Particle[NPARTICLES];
	particles_res = new Particle[NPARTICLES];

	// Allocate device memory to store the output array
	cudaMalloc(&particles_gpu, NPARTICLES*sizeof(Particle));

	for (int i = 0; i < NPARTICLES; i++)
	{
		particles_cpu[i].velocity.x = 0;
		particles_cpu[i].velocity.y = 0;
		particles_cpu[i].velocity.z = 0;

		particles_cpu[i].position.x = 1;
		particles_cpu[i].position.y = 1;
		particles_cpu[i].position.z = 1;
	}


	double iStart_HtoD = cpuSecond();
	cudaMemcpy(particles_gpu, particles_cpu, NPARTICLES*sizeof(Particle), cudaMemcpyHostToDevice);
	double iElaps_HtoD = cpuSecond() - iStart_HtoD;

	double iStart_gpu = cpuSecond();
	//GPU computation
	for (int j = 0; j < NITER; j++) 
	{
		updateKernel<<<(NPARTICLES+TPB-1) / TPB, TPB>>>(particles_gpu, dt, j);
	}
	cudaDeviceSynchronize();
	double iElaps_gpu = cpuSecond() - iStart_gpu;


	double iStart_cpu = cpuSecond();
	//CPU computation
	for (int j = 0; j < NITER; j++) 
	{
		updateCPU(particles_cpu, dt, j, NPARTICLES);
	}
	double iElaps_cpu = cpuSecond() - iStart_cpu;


	for (int i = 0; i < NPARTICLES; i++) {
		if(compareParticle(particles_cpu[i], particles_res[i])) {
			flag = 0;
			break;
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
	
	delete[] particles_cpu;
	delete[] particles_res;
	cudaFree(particles_gpu); // Free the memory

	return 0;
}