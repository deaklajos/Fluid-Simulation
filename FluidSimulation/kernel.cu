
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "helper_math.h"

#include <stdio.h>
#include <iostream>


// Kernels



__constant__ float dt = 0.1f;

__global__
void resetSimulationCUDA(const int gridResolution,
						 float2* velocityBuffer,
						 float* pressureBuffer,
						 float4* densityBuffer)
{
	uint2 id{ blockIdx.x*blockDim.x + threadIdx.x, blockIdx.y*blockDim.y + threadIdx.y };

	if(id.x < gridResolution && id.y < gridResolution)
	{
		velocityBuffer[id.x + id.y * gridResolution] = float2{ 0.0f, 0.0f };
		pressureBuffer[id.x + id.y * gridResolution] = 0.0f;
		densityBuffer[id.x + id.y * gridResolution] = float4{ 0.0f, 0.0f, 0.0f, 0.0f };
	}
}

__device__
float fract(const float x, float* b)
{
	// TODO The implementation may not be okay.
	*b = floor(x);
	return fmin(x - floor(x), 0.999999f);
}

__device__
float2 mix(float2 x, float2 y, float a)
{
	// TODO The implementation may not be okay.
	return x + (y - x) * a;
}

__device__
float4 mix(float4 x, float4 y, float a)
{
	// TODO The implementation may not be okay.
	return x + (y - x) * a;
}

// bilinear interpolation
__device__
float2 getBil(float2 p, int gridResolution, float2* buffer)
{
	p = clamp(p, make_float2(0.0f), make_float2(gridResolution));

	float2 p00 = buffer[(int)(p.x) + (int)(p.y) * gridResolution];
	float2 p10 = buffer[(int)(p.x) + 1 + (int)(p.y) * gridResolution];
	float2 p11 = buffer[(int)(p.x) + 1 + (int)(p.y + 1.0f) * gridResolution];
	float2 p01 = buffer[(int)(p.x) + (int)(p.y + 1.0f) * gridResolution];

	float flr;
	float t0 = fract(p.x, &flr);
	float t1 = fract(p.y, &flr);

	float2 v0 = mix(p00, p10, t0);
	float2 v1 = mix(p01, p11, t0);

	return mix(v0, v1, t1);
}

__device__
float4 getBil4(float2 p, int gridResolution, float4* buffer)
{
	p = clamp(p, make_float2(0.0f), make_float2(gridResolution));

	float4 p00 = buffer[(int)(p.x) + (int)(p.y) * gridResolution];
	float4 p10 = buffer[(int)(p.x) + 1 + (int)(p.y) * gridResolution];
	float4 p11 = buffer[(int)(p.x) + 1 + (int)(p.y + 1.0f) * gridResolution];
	float4 p01 = buffer[(int)(p.x) + (int)(p.y + 1.0f) * gridResolution];

	float flr;
	float t0 = fract(p.x, &flr);
	float t1 = fract(p.y, &flr);

	float4 v0 = mix(p00, p10, t0);
	float4 v1 = mix(p01, p11, t0);

	return mix(v0, v1, t1);
}

__global__
void advection(const int gridResolution,
			   float2* inputVelocityBuffer,
			   float2* outputVelocityBuffer)
{
	uint2 id{ blockIdx.x*blockDim.x + threadIdx.x, blockIdx.y*blockDim.y + threadIdx.y };

	if(id.x > 0 && id.x < gridResolution - 1 &&
	   id.y > 0 && id.y < gridResolution - 1)
	{
		float2 velocity = inputVelocityBuffer[id.x + id.y * gridResolution];

		float2 p{ (float)id.x - dt * velocity.x, (float)id.y - dt * velocity.y };

		outputVelocityBuffer[id.x + id.y * gridResolution] = getBil(p, gridResolution, inputVelocityBuffer);
	}
	else
	{
		if(id.x == 0) outputVelocityBuffer[id.x + id.y * gridResolution] = -inputVelocityBuffer[id.x + 1 + id.y * gridResolution];
		if(id.x == gridResolution - 1) outputVelocityBuffer[id.x + id.y * gridResolution] = -inputVelocityBuffer[id.x - 1 + id.y * gridResolution];
		if(id.y == 0) outputVelocityBuffer[id.x + id.y * gridResolution] = -inputVelocityBuffer[id.x + 1 + (id.y + 1) * gridResolution];
		if(id.y == gridResolution - 1) outputVelocityBuffer[id.x + id.y * gridResolution] = -inputVelocityBuffer[id.x + 1 + (id.y - 1) * gridResolution];
	}
}

__global__
void advectionDensity(const int gridResolution,
					  float2* velocityBuffer,
					  float4* inputDensityBuffer,
					  float4* outputDensityBuffer)
{
	uint2 id{ blockIdx.x*blockDim.x + threadIdx.x, blockIdx.y*blockDim.y + threadIdx.y };

	if(id.x > 0 && id.x < gridResolution - 1 &&
	   id.y > 0 && id.y < gridResolution - 1)
	{
		float2 velocity = velocityBuffer[id.x + id.y * gridResolution];

		float2 p = float2{ (float)id.x - dt * velocity.x, (float)id.y - dt * velocity.y };

		outputDensityBuffer[id.x + id.y * gridResolution] = getBil4(p, gridResolution, inputDensityBuffer);
	}
	else
	{
		outputDensityBuffer[id.x + id.y * gridResolution] = { 0.0f,  0.0f,  0.0f,  0.0f };
	}
}

__global__
void diffusion(const int gridResolution,
			   float2* inputVelocityBuffer,
			   float2* outputVelocityBuffer)
{
	uint2 id{ blockIdx.x*blockDim.x + threadIdx.x, blockIdx.y*blockDim.y + threadIdx.y };

	float viscousity = 0.01f;
	float alpha = 1.0f / (viscousity * dt);
	float beta = 1.0f / (4.0f + alpha);

	if(id.x > 0 && id.x < gridResolution - 1 &&
	   id.y > 0 && id.y < gridResolution - 1)
	{
		float2 vL = inputVelocityBuffer[id.x - 1 + id.y * gridResolution];
		float2 vR = inputVelocityBuffer[id.x + 1 + id.y * gridResolution];
		float2 vB = inputVelocityBuffer[id.x + (id.y - 1) * gridResolution];
		float2 vT = inputVelocityBuffer[id.x + (id.y + 1) * gridResolution];

		float2 velocity = inputVelocityBuffer[id.x + id.y * gridResolution];

		outputVelocityBuffer[id.x + id.y * gridResolution] = (vL + vR + vB + vT + alpha * velocity) * beta;
	}
	else
	{
		outputVelocityBuffer[id.x + id.y * gridResolution] = inputVelocityBuffer[id.x + id.y * gridResolution];
	}
}

__global__
void vorticity(const int gridResolution, float2* velocityBuffer,
			   float* vorticityBuffer)
{
	uint2 id{ blockIdx.x*blockDim.x + threadIdx.x, blockIdx.y*blockDim.y + threadIdx.y };

	if(id.x > 0 && id.x < gridResolution - 1 &&
	   id.y > 0 && id.y < gridResolution - 1)
	{
		float2 vL = velocityBuffer[id.x - 1 + id.y * gridResolution];
		float2 vR = velocityBuffer[id.x + 1 + id.y * gridResolution];
		float2 vB = velocityBuffer[id.x + (id.y - 1) * gridResolution];
		float2 vT = velocityBuffer[id.x + (id.y + 1) * gridResolution];

		vorticityBuffer[id.x + id.y * gridResolution] = (vR.y - vL.y) - (vT.x - vB.x);
	}
	else
	{
		vorticityBuffer[id.x + id.y * gridResolution] = 0.0f;
	}
}

__global__
void addVorticity(const int gridResolution, float* vorticityBuffer,
				  float2* velocityBuffer)
{
	uint2 id{ blockIdx.x*blockDim.x + threadIdx.x, blockIdx.y*blockDim.y + threadIdx.y };

	const float scale = 0.2f;

	if(id.x > 0 && id.x < gridResolution - 1 &&
	   id.y > 0 && id.y < gridResolution - 1)
	{
		float vL = vorticityBuffer[id.x - 1 + id.y * gridResolution];
		float vR = vorticityBuffer[id.x + 1 + id.y * gridResolution];
		float vB = vorticityBuffer[id.x + (id.y - 1) * gridResolution];
		float vT = vorticityBuffer[id.x + (id.y + 1) * gridResolution];

		float4 gradV{ vR - vL, vT - vB, 0.0f, 0.0f };
		float4 z{ 0.0f, 0.0f, 1.0f, 0.0f };

		if(dot(gradV, gradV))
		{
			float4 vorticityForce = make_float4(scale * cross(make_float3(gradV), make_float3(z)));
			velocityBuffer[id.x + id.y * gridResolution] += make_float2(vorticityForce.x, vorticityForce.y) * dt;
		}
	}
}

__global__
void divergence(const int gridResolution, float2* velocityBuffer,
				float* divergenceBuffer)
{
	uint2 id{ blockIdx.x*blockDim.x + threadIdx.x, blockIdx.y*blockDim.y + threadIdx.y };

	if(id.x > 0 && id.x < gridResolution - 1 &&
	   id.y > 0 && id.y < gridResolution - 1)
	{
		float2 vL = velocityBuffer[id.x - 1 + id.y * gridResolution];
		float2 vR = velocityBuffer[id.x + 1 + id.y * gridResolution];
		float2 vB = velocityBuffer[id.x + (id.y - 1) * gridResolution];
		float2 vT = velocityBuffer[id.x + (id.y + 1) * gridResolution];

		divergenceBuffer[id.x + id.y * gridResolution] = 0.5f * ((vR.x - vL.x) + (vT.y - vB.y));
	}
	else
	{
		divergenceBuffer[id.x + id.y * gridResolution] = 0.0f;
	}
}

__global__
void pressureJacobi(const int gridResolution,
					float* inputPressureBuffer,
					float* outputPressureBuffer,
					float* divergenceBuffer)
{
	uint2 id{ blockIdx.x*blockDim.x + threadIdx.x, blockIdx.y*blockDim.y + threadIdx.y };

	if(id.x > 0 && id.x < gridResolution - 1 &&
	   id.y > 0 && id.y < gridResolution - 1)
	{

		float alpha = -1.0f;
		float beta = 0.25f;

		float vL = inputPressureBuffer[id.x - 1 + id.y * gridResolution];
		float vR = inputPressureBuffer[id.x + 1 + id.y * gridResolution];
		float vB = inputPressureBuffer[id.x + (id.y - 1) * gridResolution];
		float vT = inputPressureBuffer[id.x + (id.y + 1) * gridResolution];

		float divergence = divergenceBuffer[id.x + id.y * gridResolution];

		outputPressureBuffer[id.x + id.y * gridResolution] = (vL + vR + vB + vT + alpha * divergence) * beta;
	}
	else
	{
		if(id.x == 0) outputPressureBuffer[id.x + id.y * gridResolution] = inputPressureBuffer[id.x + 1 + id.y * gridResolution];
		if(id.x == gridResolution - 1) outputPressureBuffer[id.x + id.y * gridResolution] = inputPressureBuffer[id.x - 1 + id.y * gridResolution];
		if(id.y == 0) outputPressureBuffer[id.x + id.y * gridResolution] = inputPressureBuffer[id.x + (id.y + 1) * gridResolution];
		if(id.y == gridResolution - 1) outputPressureBuffer[id.x + id.y * gridResolution] = inputPressureBuffer[id.x + (id.y - 1) * gridResolution];
	}
}

__global__
void projectionCUDA(const int gridResolution,
					float2* inputVelocityBuffer,
					float* pressureBuffer,
					float2* outputVelocityBuffer)
{
	uint2 id{ blockIdx.x*blockDim.x + threadIdx.x, blockIdx.y*blockDim.y + threadIdx.y };

	if(id.x > 0 && id.x < gridResolution - 1 &&
	   id.y > 0 && id.y < gridResolution - 1)
	{
		float pL = pressureBuffer[id.x - 1 + id.y * gridResolution];
		float pR = pressureBuffer[id.x + 1 + id.y * gridResolution];
		float pB = pressureBuffer[id.x + (id.y - 1) * gridResolution];
		float pT = pressureBuffer[id.x + (id.y + 1) * gridResolution];

		float2 velocity = inputVelocityBuffer[id.x + id.y * gridResolution];

		outputVelocityBuffer[id.x + id.y * gridResolution] = velocity -  /* 0.5f **//* (1.0f / 256.0f) **/ float2{ pR - pL, pT - pB };
	}
	else
	{
		if(id.x == 0) outputVelocityBuffer[id.x + id.y * gridResolution] = -inputVelocityBuffer[id.x + 1 + id.y * gridResolution];
		if(id.x == gridResolution - 1) outputVelocityBuffer[id.x + id.y * gridResolution] = -inputVelocityBuffer[id.x - 1 + id.y * gridResolution];
		if(id.y == 0) outputVelocityBuffer[id.x + id.y * gridResolution] = -inputVelocityBuffer[id.x + 1 + (id.y + 1) * gridResolution];
		if(id.y == gridResolution - 1) outputVelocityBuffer[id.x + id.y * gridResolution] = -inputVelocityBuffer[id.x + 1 + (id.y - 1) * gridResolution];
	}
}

__global__
void addForce(const float x, const float y, const float2 force,
			  const int gridResolution, float2* velocityBuffer,
			  const float4 density, float4* densityBuffer)
{
	uint2 id{ blockIdx.x*blockDim.x + threadIdx.x, blockIdx.y*blockDim.y + threadIdx.y };

	float dx = ((float)id.x / (float)gridResolution) - x;
	float dy = ((float)id.y / (float)gridResolution) - y;

	float radius = 0.001f;

	float c = exp(-(dx * dx + dy * dy) / radius) * dt;

	velocityBuffer[id.x + id.y * gridResolution] += c * force;
	densityBuffer[id.x + id.y * gridResolution] += c * density;
}

// *************
// Visualization
// *************

__global__
void visualizationDensity(const int width, const int height, float4* visualizationBuffer,
						  const int gridResolution, float4* densityBuffer)
{
	uint2 id{ blockIdx.x*blockDim.x + threadIdx.x, blockIdx.y*blockDim.y + threadIdx.y };

	if(id.x < width && id.y < height)
	{
		float4 density = densityBuffer[id.x + id.y * width];
		visualizationBuffer[id.x + id.y * width] = density;
	}
}

__global__
void visualizationVelocity(const int width, const int height, float4* visualizationBuffer,
						   const int gridResolution, float2* velocityBuffer)
{
	uint2 id{ blockIdx.x*blockDim.x + threadIdx.x, blockIdx.y*blockDim.y + threadIdx.y };

	if(id.x < width && id.y < height)
	{
		float2 velocity = velocityBuffer[id.x + id.y * width];
		float2 tmp = (1.0f + velocity) / 2.0f;
		visualizationBuffer[id.x + id.y * width] = float4{ tmp.x, tmp.y, 0.0f, 0.0f };
	}
}

__global__
void visualizationPressure(const int width, const int height, float4* visualizationBuffer,
						   const int gridResolution, float* pressureBuffer)
{
	uint2 id{ blockIdx.x*blockDim.x + threadIdx.x, blockIdx.y*blockDim.y + threadIdx.y };

	if(id.x < width && id.y < height)
	{
		float pressure = pressureBuffer[id.x + id.y * width];
		visualizationBuffer[id.x + id.y * width] = float4{ (1.0f + pressure) / 2.0f }; // TODO somtin???
	}
}

// End of kernels

// Buffers

// simulation
int gridResolution = 512;
dim3 threadsPerBlock(gridResolution, gridResolution);

int inputVelocityBuffer = 0;
float2* velocityBuffer[2];

int inputDensityBuffer = 0;
float4* densityBuffer[2];
float4 densityColor;

int inputPressureBuffer = 0;
float* pressureBuffer[2];
float* divergenceBuffer;

float* vorticityBuffer;

size_t problemSize[2];

float2 force;

// visualization
int width = 512;
int height = 512;

float4* visualizationBufferGPU;
float4* visualizationBufferCPU;

int visualizationMethod = 0;

size_t visualizationSize[2];

// End of Buffers

void initBuffers()
{
	problemSize[0] = gridResolution;
	problemSize[1] = gridResolution;

	cudaMalloc(&velocityBuffer[0], sizeof(float2) * gridResolution * gridResolution);
	cudaMalloc(&velocityBuffer[1], sizeof(float2) * gridResolution * gridResolution);

	cudaMalloc(&densityBuffer[0], sizeof(float4) * gridResolution * gridResolution);
	cudaMalloc(&densityBuffer[1], sizeof(float4) * gridResolution * gridResolution);

	cudaMalloc(&pressureBuffer[0], sizeof(float) * gridResolution * gridResolution);
	cudaMalloc(&pressureBuffer[1], sizeof(float) * gridResolution * gridResolution);

	cudaMalloc(&divergenceBuffer, sizeof(float) * gridResolution * gridResolution);

	cudaMalloc(&vorticityBuffer, sizeof(float) * gridResolution * gridResolution);

	// TODO Could be different.
	densityColor = float4{ 1.0f };

	visualizationSize[0] = width;
	visualizationSize[1] = height;

	visualizationBufferCPU = new float4[width * height];
	cudaMalloc(&visualizationBufferGPU, sizeof(float4) * width * height);
}

void resetSimulation()
{
	resetSimulationCUDA<<<1, threadsPerBlock>>>(gridResolution,
													 velocityBuffer[inputVelocityBuffer],
													 pressureBuffer[inputPressureBuffer],
													 densityBuffer[inputDensityBuffer]);
}

void resetPressure()
{
	resetSimulationCUDA << <1, threadsPerBlock >> > (gridResolution,
													 velocityBuffer[(inputVelocityBuffer + 1) % 2],
													 pressureBuffer[inputPressureBuffer],
													 densityBuffer[(inputDensityBuffer + 1) % 2]);
}

void simulateAdvection()
{
	int nextBufferIndex = (inputVelocityBuffer + 1) % 2;
	advection << <1, threadsPerBlock >> > (gridResolution,
										   velocityBuffer[inputVelocityBuffer],
										   velocityBuffer[nextBufferIndex]);
	inputVelocityBuffer = nextBufferIndex;
}

void simulateVorticity()
{
	vorticity << <1, threadsPerBlock >> > (gridResolution,
										   velocityBuffer[inputVelocityBuffer],
										   vorticityBuffer);

	addVorticity << <1, threadsPerBlock >> > (gridResolution,
											  vorticityBuffer,
											  velocityBuffer[inputVelocityBuffer]);
}

void simulateDiffusion()
{
	for(int i = 0; i < 10; ++i)
	{
		int nextBufferIndex = (inputVelocityBuffer + 1) % 2;
		diffusion << <1, threadsPerBlock >> > (gridResolution,
											   velocityBuffer[inputVelocityBuffer],
											   velocityBuffer[nextBufferIndex]);
		inputVelocityBuffer = nextBufferIndex;
	}
}

void projection()
{
	divergence << <1, threadsPerBlock >> > (gridResolution,
											velocityBuffer[inputVelocityBuffer],
											divergenceBuffer);

	resetPressure();

	for(int i = 0; i < 10; ++i)
	{
		int nextBufferIndex = (inputPressureBuffer + 1) % 2;
		pressureJacobi << <1, threadsPerBlock >> > (gridResolution,
											   pressureBuffer[inputPressureBuffer],
											   pressureBuffer[nextBufferIndex],
											   divergenceBuffer);
		inputPressureBuffer = nextBufferIndex;
	}

	int nextBufferIndex = (inputVelocityBuffer + 1) % 2;
	projectionCUDA << <1, threadsPerBlock >> > (gridResolution,
												velocityBuffer[inputVelocityBuffer],
												pressureBuffer[inputPressureBuffer],
												velocityBuffer[nextBufferIndex]);
	inputVelocityBuffer = nextBufferIndex;
}

void simulateDensityAdvection()
{
	int nextBufferIndex = (inputVelocityBuffer + 1) % 2;
	advectionDensity << <1, threadsPerBlock >> > (gridResolution,
												  velocityBuffer[inputVelocityBuffer],
												  densityBuffer[inputDensityBuffer],
												  densityBuffer[nextBufferIndex]);

	inputVelocityBuffer = nextBufferIndex;
}

void addForce(int x, int y, float2 force)
{
	float fx = (float)x / width;
	float fy = (float)y / height;

	addForce << <1, threadsPerBlock >> > (fx, fy, force, gridResolution,
										  velocityBuffer[inputVelocityBuffer],
										  densityColor, densityBuffer[inputDensityBuffer]);
}

void simulationStep()
{
	simulateAdvection();
	simulateDiffusion();
	simulateVorticity();
	projection();
	simulateDensityAdvection();
}

void visualizationStep()
{
	switch(visualizationMethod)
	{
	case 0:
		visualizationDensity << <1, threadsPerBlock >> > (width, height,
														  visualizationBufferGPU,
														  gridResolution,
														  densityBuffer[inputDensityBuffer]);
		break;

	case 1:
		visualizationVelocity << <1, threadsPerBlock >> > (width, height,
														  visualizationBufferGPU,
														  gridResolution,
														  velocityBuffer[inputVelocityBuffer]);
		break;

	case 2:
		visualizationPressure << <1, threadsPerBlock >> > (width, height,
														  visualizationBufferGPU,
														  gridResolution,
														  pressureBuffer[inputPressureBuffer]);
		break;
	}

	cudaMemcpy(visualizationBufferCPU, visualizationBufferGPU, sizeof(float4) * width * height, cudaMemcpyDeviceToHost);
	// TODO Draw PIXELS
}

// TODO OPENGL

int main()
{


	// cudaDeviceReset must be called before exiting in order for profiling and
	// tracing tools such as Nsight and Visual Profiler to show complete traces.
	cudaError_t cudaStatus = cudaDeviceReset();
	if(cudaStatus != cudaSuccess)
	{
		fprintf(stderr, "cudaDeviceReset failed!");
		return 1;
	}

	return 0;
}
