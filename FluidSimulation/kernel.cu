
#include "cuda_runtime.h"
#include "cuda_runtime_api.h"
#include "device_launch_parameters.h"
#include "helper_math.h"
#include "surface_functions.h"
#include <helper_cuda.h>
#include <helper_functions.h> 

#include <stdio.h>
#include <iostream>

#include <GL/glew.h>
#include <GL/freeglut.h>

#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort = true)
{
	if(code != cudaSuccess)
	{
		fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
		if(abort) exit(code);
	}
}

// Kernels
texture<float, 2, cudaReadModeElementType> texture_float_1;
texture<float, 2, cudaReadModeElementType> texture_float_2;
texture<float2, 2, cudaReadModeElementType> texture_float2;
surface<void, 2> surface_out_1;
surface<void, 2> surface_out_2;
surface<void, 2> surface_out_3;
texture<float4, 2, cudaReadModeElementType> texture_float4;
cudaChannelFormatDesc desc_float = cudaCreateChannelDesc<float>();
cudaChannelFormatDesc desc_float2 = cudaCreateChannelDesc<float2>();
cudaChannelFormatDesc desc_float4 = cudaCreateChannelDesc<float4>();

__constant__ float dt = 0.1f;
__device__ int cnt = 0;

__global__
void resetSimulationCUDA(const int gridResolution,
						 float2* velocityBuffer,
						 float* pressureBuffer,
						 float4* densityBuffer)
{
	
	uint2 id{ blockIdx.x*blockDim.x + threadIdx.x, blockIdx.y*blockDim.y + threadIdx.y };

	if(id.x < gridResolution && id.y < gridResolution)
	{
		surf2Dwrite(float2{ 0.0f, 0.0f }, surface_out_1, id.x * sizeof(float2), id.y);
		surf2Dwrite(0.0f, surface_out_2, id.x * sizeof(float), id.y);
		surf2Dwrite(float4{ 0.0f, 0.0f, 0.0f, 0.0f }, surface_out_3, id.x * sizeof(float4), id.y);

		//velocityBuffer[id.x + id.y * gridResolution] = float2{ 0.0f, 0.0f };
		//pressureBuffer[id.x + id.y * gridResolution] = 0.0f;
		//densityBuffer[id.x + id.y * gridResolution] = float4{ 0.0f, 0.0f, 0.0f, 0.0f };
	}
}

__device__
float fract(const float x, float* b)
{
	// TODO The implementation may not be okay.
	*b = floor(x);
	return fmin(x - floor(x), 0.999f);
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
		float2 velocity = tex2D(texture_float2, id.x + 0.5f, id.y + 0.5f);
		float2 p = make_float2((float)id.x - dt * velocity.x, (float)id.y - dt * velocity.y);

		//TODO think: set bordertype
		//TODO get it and write it after
		p = clamp(p, make_float2(0.0f), make_float2(gridResolution));
		surf2Dwrite(tex2D(texture_float2, p.x + 0.5f, p.y + 0.5f), surface_out_1, id.x * sizeof(float2), id.y);
	}
	else
	{
		if(id.x == 0) surf2Dwrite(-tex2D(texture_float2, id.x + 1, id.y), surface_out_1, id.x * sizeof(float2), id.y);
		if(id.x == gridResolution - 1) surf2Dwrite(-tex2D(texture_float2, id.x - 1, id.y), surface_out_1, id.x * sizeof(float2), id.y);
		if(id.y == 0) surf2Dwrite(-tex2D(texture_float2, id.x + 1, id.y + 1), surface_out_1, id.x * sizeof(float2), id.y);
		if(id.y == gridResolution - 1) surf2Dwrite(-tex2D(texture_float2, id.x + 1, id.y - 1), surface_out_1, id.x * sizeof(float2), id.y);
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
		/*float2 velocity = velocityBuffer[id.x + id.y * gridResolution];

		float2 p = float2{ (float)id.x - dt * velocity.x, (float)id.y - dt * velocity.y };

		outputDensityBuffer[id.x + id.y * gridResolution] = getBil4(p, gridResolution, inputDensityBuffer);*/



		float2 velocity = tex2D(texture_float2, id.x + 0.5f, id.y + 0.5f);

		float2 p = float2{ (float)id.x - dt * velocity.x, (float)id.y - dt * velocity.y };

		p = clamp(p, make_float2(0.0f), make_float2(gridResolution));
		surf2Dwrite(tex2D(texture_float4, p.x + 0.5f, p.y + 0.5f), surface_out_1, id.x * sizeof(float4), id.y);

	}
	else
	{
		//outputDensityBuffer[id.x + id.y * gridResolution] = float4{ 0.0f,  0.0f,  0.0f,  0.0f };

		surf2Dwrite(float4{ 0.0f,  0.0f,  0.0f,  0.0f }, surface_out_1, id.x * sizeof(float4), id.y);
	}
}

//TODO remove debug function
__global__ void myprint()
{
	printf("[%d, %d]\n", blockIdx.y*gridDim.x + blockIdx.x, blockIdx.y*gridDim.y + blockIdx.y);
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
		//float2 vL = inputVelocityBuffer[id.x - 1 + id.y * gridResolution];
		//float2 vR = inputVelocityBuffer[id.x + 1 + id.y * gridResolution];
		//float2 vB = inputVelocityBuffer[id.x + (id.y - 1) * gridResolution];
		//float2 vT = inputVelocityBuffer[id.x + (id.y + 1) * gridResolution];

		//float2 velocity = inputVelocityBuffer[id.x + id.y * gridResolution];

		//outputVelocityBuffer[id.x + id.y * gridResolution] = (vL + vR + vB + vT + alpha * velocity) * beta;

		//tex
		float2 vL = tex2D(texture_float2, id.x - 1 + 0.5f, id.y + 0.5f);
		float2 vR = tex2D(texture_float2, id.x + 1 + 0.5f, id.y + 0.5f);
		float2 vB = tex2D(texture_float2, id.x + 0.5f, id.y - 1 + 0.5f);
		float2 vT = tex2D(texture_float2, id.x + 0.5f, id.y + 1 + 0.5f);

		float2 velocity = tex2D(texture_float2, id.x + 0.5f, id.y + 0.5f);

		float2 out = (vL + vR + vB + vT + alpha * velocity) * beta;

		surf2Dwrite(out, surface_out_1, id.x * sizeof(float2), id.y);
	}
	else
	{
		/*outputVelocityBuffer[id.x + id.y * gridResolution] = inputVelocityBuffer[id.x + id.y * gridResolution];*/

		float2 velocity = tex2D(texture_float2, id.x + 0.5f, id.y + 0.5f);

		surf2Dwrite(velocity, surface_out_1, id.x * sizeof(float2), id.y);
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
		//float2 vL = velocityBuffer[id.x - 1 + id.y * gridResolution];
		//float2 vR = velocityBuffer[id.x + 1 + id.y * gridResolution];
		//float2 vB = velocityBuffer[id.x + (id.y - 1) * gridResolution];
		//float2 vT = velocityBuffer[id.x + (id.y + 1) * gridResolution];

		//vorticityBuffer[id.x + id.y * gridResolution] = (vR.y - vL.y) - (vT.x - vB.x);


		float2 vL = tex2D(texture_float2, id.x - 1 + 0.5f, id.y + 0.5f);
		float2 vR = tex2D(texture_float2, id.x + 1 + 0.5f, id.y + 0.5f);
		float2 vB = tex2D(texture_float2, id.x + 0.5f, id.y - 1 + 0.5f);
		float2 vT = tex2D(texture_float2, id.x + 0.5f, id.y + 1 + 0.5f);

		float out = (vR.y - vL.y) - (vT.x - vB.x);

		surf2Dwrite(out, surface_out_1, id.x * sizeof(float), id.y);
	}
	else
	{
		//vorticityBuffer[id.x + id.y * gridResolution] = 0.0f;

		surf2Dwrite(0.0f, surface_out_1, id.x * sizeof(float), id.y);
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
		//float vL = vorticityBuffer[id.x - 1 + id.y * gridResolution];
		//float vR = vorticityBuffer[id.x + 1 + id.y * gridResolution];
		//float vB = vorticityBuffer[id.x + (id.y - 1) * gridResolution];
		//float vT = vorticityBuffer[id.x + (id.y + 1) * gridResolution];

		//float4 gradV{ vR - vL, vT - vB, 0.0f, 0.0f };
		//float4 z{ 0.0f, 0.0f, 1.0f, 0.0f };

		//if(dot(gradV, gradV))
		//{
		//	float4 vorticityForce = make_float4(scale * cross(make_float3(gradV), make_float3(z)));
		//	velocityBuffer[id.x + id.y * gridResolution] += make_float2(vorticityForce.x, vorticityForce.y) * dt;
		//}


		float vL = tex2D(texture_float_1, id.x - 1 + 0.5f, id.y + 0.5f);
		float vR = tex2D(texture_float_1, id.x + 1 + 0.5f, id.y + 0.5f);
		float vB = tex2D(texture_float_1, id.x + 0.5f, id.y - 1 + 0.5f);
		float vT = tex2D(texture_float_1, id.x + 0.5f, id.y + 1 + 0.5f);

		float4 gradV{ vR - vL, vT - vB, 0.0f, 0.0f };
		float4 z{ 0.0f, 0.0f, 1.0f, 0.0f };

		if(dot(gradV, gradV))
		{
			float4 vorticityForce = make_float4(scale * cross(make_float3(gradV), make_float3(z)));

			float2 temp;
			surf2Dread(&temp, surface_out_1, id.x * sizeof(float2), id.y);

			temp += make_float2(vorticityForce.x, vorticityForce.y) * dt;
			surf2Dwrite(temp, surface_out_1, id.x * sizeof(float2), id.y);
		}
	}
}

__global__
void divergence(const int gridResolution, float2* velocityBuffer,
				float* divergenceBuffer)
{
	
	uint2 id{ blockIdx.x*blockDim.x + threadIdx.x, blockIdx.y*blockDim.y + threadIdx.y };


	//TODO border would solve this.
	if(id.x > 0 && id.x < gridResolution - 1 &&
	   id.y > 0 && id.y < gridResolution - 1)
	{
		//float2 vL = velocityBuffer[id.x - 1 + id.y * gridResolution];
		//float2 vR = velocityBuffer[id.x + 1 + id.y * gridResolution];
		//float2 vB = velocityBuffer[id.x + (id.y - 1) * gridResolution];
		//float2 vT = velocityBuffer[id.x + (id.y + 1) * gridResolution];

		//divergenceBuffer[id.x + id.y * gridResolution] = 0.5f * ((vR.x - vL.x) + (vT.y - vB.y));

		float2 vL = tex2D(texture_float2, id.x - 1 + 0.5f, id.y + 0.5f);
		float2 vR = tex2D(texture_float2, id.x + 1 + 0.5f, id.y + 0.5f);
		float2 vB = tex2D(texture_float2, id.x + 0.5f, id.y - 1 + 0.5f);
		float2 vT = tex2D(texture_float2, id.x + 0.5f, id.y + 1 + 0.5f);

		float out = 0.5f * ((vR.x - vL.x) + (vT.y - vB.y));
		surf2Dwrite(out, surface_out_1, id.x * sizeof(float), id.y);
	}
	else
	{
		//divergenceBuffer[id.x + id.y * gridResolution] = 0.0f;

		surf2Dwrite(0.0f, surface_out_1, id.x * sizeof(float), id.y);
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

		//float vL = inputPressureBuffer[id.x - 1 + id.y * gridResolution];
		//float vR = inputPressureBuffer[id.x + 1 + id.y * gridResolution];
		//float vB = inputPressureBuffer[id.x + (id.y - 1) * gridResolution];
		//float vT = inputPressureBuffer[id.x + (id.y + 1) * gridResolution];

		//float divergence = divergenceBuffer[id.x + id.y * gridResolution];

		//outputPressureBuffer[id.x + id.y * gridResolution] = (vL + vR + vB + vT + alpha * divergence) * beta;


		float vL = tex2D(texture_float_1, id.x - 1 + 0.5f, id.y + 0.5f);
		float vR = tex2D(texture_float_1, id.x + 1 + 0.5f, id.y + 0.5f);
		float vB = tex2D(texture_float_1, id.x + 0.5f, id.y - 1 + 0.5f);
		float vT = tex2D(texture_float_1, id.x + 0.5f, id.y + 1 + 0.5f);

		float divergence = tex2D(texture_float_2, id.x + 0.5f, id.y + 0.5f);

		float out = (vL + vR + vB + vT + alpha * divergence) * beta;

		surf2Dwrite(out, surface_out_1, id.x * sizeof(float), id.y);
	}
	else
	{
		//if(id.x == 0) outputPressureBuffer[id.x + id.y * gridResolution] = inputPressureBuffer[id.x + 1 + id.y * gridResolution];
		//if(id.x == gridResolution - 1) outputPressureBuffer[id.x + id.y * gridResolution] = inputPressureBuffer[id.x - 1 + id.y * gridResolution];
		//if(id.y == 0) outputPressureBuffer[id.x + id.y * gridResolution] = inputPressureBuffer[id.x + (id.y + 1) * gridResolution];
		//if(id.y == gridResolution - 1) outputPressureBuffer[id.x + id.y * gridResolution] = inputPressureBuffer[id.x + (id.y - 1) * gridResolution];

		if(id.x == 0) surf2Dwrite(-tex2D(texture_float_1, id.x + 1, id.y), surface_out_1, id.x * sizeof(float), id.y);
		if(id.x == gridResolution - 1) surf2Dwrite(-tex2D(texture_float_1, id.x - 1, id.y), surface_out_1, id.x * sizeof(float), id.y);
		if(id.y == 0) surf2Dwrite(-tex2D(texture_float_1, id.x, id.y + 1), surface_out_1, id.x * sizeof(float), id.y);
		if(id.y == gridResolution - 1) surf2Dwrite(-tex2D(texture_float_1, id.x, id.y - 1), surface_out_1, id.x * sizeof(float), id.y);
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
		//float pL = pressureBuffer[id.x - 1 + id.y * gridResolution];
		//float pR = pressureBuffer[id.x + 1 + id.y * gridResolution];
		//float pB = pressureBuffer[id.x + (id.y - 1) * gridResolution];
		//float pT = pressureBuffer[id.x + (id.y + 1) * gridResolution];

		//float2 velocity = inputVelocityBuffer[id.x + id.y * gridResolution];

		//outputVelocityBuffer[id.x + id.y * gridResolution] = velocity - float2{ pR - pL, pT - pB };


		float pL = tex2D(texture_float_1, id.x - 1 + 0.5f, id.y + 0.5f);
		float pR = tex2D(texture_float_1, id.x + 1 + 0.5f, id.y + 0.5f);
		float pB = tex2D(texture_float_1, id.x + 0.5f, id.y - 1 + 0.5f);
		float pT = tex2D(texture_float_1, id.x + 0.5f, id.y + 1 + 0.5f);

		float2 velocity = tex2D(texture_float2, id.x + 0.5f, id.y + 0.5f);
		float2 out = velocity - float2{ pR - pL, pT - pB };

		surf2Dwrite(out, surface_out_1, id.x * sizeof(float2), id.y);
	}
	else
	{
		//if(id.x == 0) outputVelocityBuffer[id.x + id.y * gridResolution] = -inputVelocityBuffer[id.x + 1 + id.y * gridResolution];
		//if(id.x == gridResolution - 1) outputVelocityBuffer[id.x + id.y * gridResolution] = -inputVelocityBuffer[id.x - 1 + id.y * gridResolution];
		//if(id.y == 0) outputVelocityBuffer[id.x + id.y * gridResolution] = -inputVelocityBuffer[id.x + 1 + (id.y + 1) * gridResolution];
		//if(id.y == gridResolution - 1) outputVelocityBuffer[id.x + id.y * gridResolution] = -inputVelocityBuffer[id.x + 1 + (id.y - 1) * gridResolution];

		if(id.x == 0) surf2Dwrite(-tex2D(texture_float2, id.x + 1, id.y), surface_out_1, id.x * sizeof(float2), id.y);
		if(id.x == gridResolution - 1) surf2Dwrite(-tex2D(texture_float2, id.x - 1, id.y), surface_out_1, id.x * sizeof(float2), id.y);
		if(id.y == 0) surf2Dwrite(-tex2D(texture_float2, id.x + 1, id.y + 1), surface_out_1, id.x * sizeof(float2), id.y);
		if(id.y == gridResolution - 1) surf2Dwrite(-tex2D(texture_float2, id.x + 1, id.y - 1), surface_out_1, id.x * sizeof(float2), id.y);
	}
}

__global__
void addForceCUDA(const float x, const float y, const float2 force,
			  const int gridResolution, float2* velocityBuffer,
			  const float4 density, float4* densityBuffer)
{
	
	uint2 id{ blockIdx.x*blockDim.x + threadIdx.x, blockIdx.y*blockDim.y + threadIdx.y };

	float dx = ((float)id.x / (float)gridResolution) - x;
	float dy = ((float)id.y / (float)gridResolution) - y;

	float radius = 0.001f;

	float c = exp(-(dx * dx + dy * dy) / radius) * dt;


	//velocityBuffer[id.x + id.y * gridResolution] += c * force;

	//TODO remove variable
	cnt = 0;


	float2 temp2;
	surf2Dread(&temp2, surface_out_1, id.x * sizeof(float2), id.y);
	surf2Dwrite(temp2 + c * force, surface_out_1, id.x * sizeof(float2), id.y);

	//if(fabs((temp + c * force).x - velocityBuffer[id.x + id.y * gridResolution].x) > 0.00001)
	//	temp.x = temp.x;
	float4 temp4;
	surf2Dread(&temp4, surface_out_2, id.x * sizeof(float4), id.y);
	surf2Dwrite(temp4 + c * density, surface_out_2, id.x * sizeof(float4), id.y);
	//densityBuffer[id.x + id.y * gridResolution] += c * density;
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
		float4 density = tex2D(texture_float4, id.x, id.y);
		//float4 density = densityBuffer[id.x + id.y * width];
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
		//float2 velocity = velocityBuffer[id.x + id.y * width];
		float2 velocity = tex2D(texture_float2, id.x, id.y);
		//if(velocity.x > 0.0001f)
		//	velocity.x = 0;
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
		//float pressure = pressureBuffer[id.x + id.y * width];
		float pressure = tex2D(texture_float_1, id.x, id.y);

		visualizationBuffer[id.x + id.y * width] = make_float4((1.0f + pressure) / 2.0f ); // TODO somtin???
	}
}

// End of kernels

// Buffers

// simulation
int gridResolution = 512;
dim3 threadsPerBlock(32, 32);
dim3 numBlocks(gridResolution / threadsPerBlock.x, gridResolution / threadsPerBlock.y);

int inputVelocityBuffer = 0;
float2* velocityBuffer[2];
cudaArray* velocityBufferArray[2];

int inputDensityBuffer = 0;
float4* densityBuffer[2];
cudaArray* densityBufferArray[2];
float4 densityColor;

int inputPressureBuffer = 0;
float* pressureBuffer[2];
cudaArray* pressureBufferArray[2];

float* divergenceBuffer;
cudaArray* divergenceBufferArray;

float* vorticityBuffer;
cudaArray* vorticityBufferArray;

size_t problemSize[2];

float2 force;

// visualization
int width = 512;
int height = 512;

float4* visualizationBufferGPU;
cudaArray* visualizationBufferArrayGPU;
float4* visualizationBufferCPU;

int visualizationMethod = 0;

size_t visualizationSize[2];

// End of Buffers
void addForce(int x, int y, float2 force);
void resetSimulation();

void initBuffers()
{
	problemSize[0] = gridResolution;
	problemSize[1] = gridResolution;

	gpuErrchk(cudaMalloc(&velocityBuffer[0], sizeof(float2) * gridResolution * gridResolution));
	gpuErrchk(cudaMalloc(&velocityBuffer[1], sizeof(float2) * gridResolution * gridResolution));
	gpuErrchk(cudaMallocArray(&velocityBufferArray[0], &desc_float2, gridResolution, gridResolution, cudaArraySurfaceLoadStore));
	gpuErrchk(cudaMallocArray(&velocityBufferArray[1], &desc_float2, gridResolution, gridResolution, cudaArraySurfaceLoadStore));

	gpuErrchk(cudaMalloc(&densityBuffer[0], sizeof(float4) * gridResolution * gridResolution));
	gpuErrchk(cudaMalloc(&densityBuffer[1], sizeof(float4) * gridResolution * gridResolution));
	gpuErrchk(cudaMallocArray(&densityBufferArray[0], &desc_float4, gridResolution, gridResolution, cudaArraySurfaceLoadStore));
	gpuErrchk(cudaMallocArray(&densityBufferArray[1], &desc_float4, gridResolution, gridResolution, cudaArraySurfaceLoadStore));

	gpuErrchk(cudaMalloc(&pressureBuffer[0], sizeof(float) * gridResolution * gridResolution));
	gpuErrchk(cudaMalloc(&pressureBuffer[1], sizeof(float) * gridResolution * gridResolution));
	gpuErrchk(cudaMallocArray(&pressureBufferArray[0], &desc_float, gridResolution, gridResolution, cudaArraySurfaceLoadStore));
	gpuErrchk(cudaMallocArray(&pressureBufferArray[1], &desc_float, gridResolution, gridResolution, cudaArraySurfaceLoadStore));

	gpuErrchk(cudaMalloc(&divergenceBuffer, sizeof(float) * gridResolution * gridResolution));
	gpuErrchk(cudaMallocArray(&divergenceBufferArray, &desc_float, gridResolution, gridResolution, cudaArraySurfaceLoadStore));

	gpuErrchk(cudaMalloc(&vorticityBuffer, sizeof(float) * gridResolution * gridResolution));
	gpuErrchk(cudaMallocArray(&vorticityBufferArray, &desc_float, gridResolution, gridResolution, cudaArraySurfaceLoadStore));

	densityColor = float4{ 1.0f, 1.0f, 1.0f, 1.0f };

	visualizationSize[0] = width;
	visualizationSize[1] = height;

	gpuErrchk(cudaMallocHost((void**)&visualizationBufferCPU, sizeof(float4) * width * height));
	gpuErrchk(cudaMalloc(&visualizationBufferGPU, sizeof(float4) * width * height));
	gpuErrchk(cudaMallocArray(&visualizationBufferArrayGPU, &desc_float4, gridResolution, gridResolution, cudaArraySurfaceLoadStore));
	resetSimulation();
}

void resetSimulation()
{
	gpuErrchk(cudaBindSurfaceToArray(surface_out_1, velocityBufferArray[inputVelocityBuffer]));
	gpuErrchk(cudaBindSurfaceToArray(surface_out_2, pressureBufferArray[inputPressureBuffer]));
	gpuErrchk(cudaBindSurfaceToArray(surface_out_3, densityBufferArray[inputDensityBuffer]));
	resetSimulationCUDA<<<numBlocks, threadsPerBlock>>>(gridResolution,
													 velocityBuffer[inputVelocityBuffer],
													 pressureBuffer[inputPressureBuffer],
													 densityBuffer[inputDensityBuffer]);
	gpuErrchk(cudaPeekAtLastError());
}

void resetPressure()
{
	gpuErrchk(cudaBindSurfaceToArray(surface_out_1, velocityBufferArray[(inputVelocityBuffer + 1) % 2]));
	gpuErrchk(cudaBindSurfaceToArray(surface_out_2, pressureBufferArray[inputPressureBuffer]));
	gpuErrchk(cudaBindSurfaceToArray(surface_out_3, densityBufferArray[(inputDensityBuffer + 1) % 2]));
	resetSimulationCUDA<<<numBlocks, threadsPerBlock >> > (gridResolution,
													 velocityBuffer[(inputVelocityBuffer + 1) % 2],
													 pressureBuffer[inputPressureBuffer],
													 densityBuffer[(inputDensityBuffer + 1) % 2]);
	gpuErrchk(cudaPeekAtLastError());
}

void simulateAdvection()
{
	int nextBufferIndex = (inputVelocityBuffer + 1) % 2;
	gpuErrchk(cudaBindTextureToArray(texture_float2, velocityBufferArray[inputVelocityBuffer], desc_float2));
	texture_float2.filterMode = cudaFilterModeLinear;
	//texture_float2_in.filterMode = cudaFilterModePoint;
	gpuErrchk(cudaBindSurfaceToArray(surface_out_1, velocityBufferArray[nextBufferIndex]));

	advection<<<numBlocks, threadsPerBlock >>>(gridResolution,
										   velocityBuffer[inputVelocityBuffer],
										   velocityBuffer[nextBufferIndex]);
	gpuErrchk(cudaPeekAtLastError());
	gpuErrchk(cudaUnbindTexture(texture_float2));
	inputVelocityBuffer = nextBufferIndex;
}

void simulateVorticity()
{
	gpuErrchk(cudaBindTextureToArray(texture_float2, velocityBufferArray[inputVelocityBuffer], desc_float2));
	texture_float2.filterMode = cudaFilterModeLinear;

	gpuErrchk(cudaBindSurfaceToArray(surface_out_1, vorticityBufferArray));

	vorticity <<<numBlocks, threadsPerBlock>>> (gridResolution,
										   velocityBuffer[inputVelocityBuffer],
										   vorticityBuffer);
	gpuErrchk(cudaUnbindTexture(texture_float2));
	gpuErrchk(cudaPeekAtLastError());

	gpuErrchk(cudaBindTextureToArray(texture_float_1, vorticityBufferArray, desc_float));
	texture_float_1.filterMode = cudaFilterModeLinear;

	gpuErrchk(cudaBindSurfaceToArray(surface_out_1, velocityBufferArray[inputVelocityBuffer]));

	addVorticity <<<numBlocks, threadsPerBlock>>> (gridResolution,
											  vorticityBuffer,
											  velocityBuffer[inputVelocityBuffer]);
	gpuErrchk(cudaPeekAtLastError());
	gpuErrchk(cudaUnbindTexture(texture_float_1));
}

void simulateDiffusion()
{
	for(int i = 0; i < 10; ++i)
	{
		int nextBufferIndex = (inputVelocityBuffer + 1) % 2;

		gpuErrchk(cudaBindTextureToArray(texture_float2, velocityBufferArray[inputVelocityBuffer], desc_float2));
		//texture_float2_in.filterMode = cudaFilterModeLinear;
		texture_float2.filterMode = cudaFilterModePoint;

		gpuErrchk(cudaBindSurfaceToArray(surface_out_1, velocityBufferArray[nextBufferIndex]));
		
		diffusion <<<numBlocks, threadsPerBlock>>> (gridResolution,
											   velocityBuffer[inputVelocityBuffer],
											   velocityBuffer[nextBufferIndex]);

		gpuErrchk(cudaPeekAtLastError());
		gpuErrchk(cudaUnbindTexture(texture_float2));

		inputVelocityBuffer = nextBufferIndex;
	}
}

void projection()
{

	gpuErrchk(cudaBindTextureToArray(texture_float2, velocityBufferArray[inputVelocityBuffer], desc_float2));
	texture_float2.filterMode = cudaFilterModePoint;

	gpuErrchk(cudaBindSurfaceToArray(surface_out_1, divergenceBufferArray));

	divergence <<<numBlocks, threadsPerBlock>>> (gridResolution,
											velocityBuffer[inputVelocityBuffer],
											divergenceBuffer);
	gpuErrchk(cudaPeekAtLastError());
	gpuErrchk(cudaUnbindTexture(texture_float2));

	resetPressure();

	for(int i = 0; i < 10; ++i)
	{
		int nextBufferIndex = (inputPressureBuffer + 1) % 2;

		gpuErrchk(cudaBindTextureToArray(texture_float_1, pressureBufferArray[inputPressureBuffer], desc_float));
		texture_float_1.filterMode = cudaFilterModePoint;

		gpuErrchk(cudaBindTextureToArray(texture_float_2, divergenceBufferArray, desc_float));
		texture_float_2.filterMode = cudaFilterModePoint;

		gpuErrchk(cudaBindSurfaceToArray(surface_out_1, pressureBufferArray[nextBufferIndex]));

		pressureJacobi <<<numBlocks, threadsPerBlock>>> (gridResolution,
											   pressureBuffer[inputPressureBuffer],
											   pressureBuffer[nextBufferIndex],
											   divergenceBuffer);
		gpuErrchk(cudaPeekAtLastError());
		gpuErrchk(cudaUnbindTexture(texture_float_1));
		gpuErrchk(cudaUnbindTexture(texture_float_2));

		inputPressureBuffer = nextBufferIndex;
	}

	int nextBufferIndex = (inputVelocityBuffer + 1) % 2;

	gpuErrchk(cudaBindTextureToArray(texture_float2, velocityBufferArray[inputVelocityBuffer], desc_float2));
	texture_float2.filterMode = cudaFilterModePoint;

	gpuErrchk(cudaBindTextureToArray(texture_float_1, pressureBufferArray[inputPressureBuffer], desc_float));
	texture_float_1.filterMode = cudaFilterModePoint;

	gpuErrchk(cudaBindSurfaceToArray(surface_out_1, velocityBufferArray[nextBufferIndex]));

	projectionCUDA <<<numBlocks, threadsPerBlock>>> (gridResolution,
												velocityBuffer[inputVelocityBuffer],
												pressureBuffer[inputPressureBuffer],
												velocityBuffer[nextBufferIndex]);
	gpuErrchk(cudaPeekAtLastError());
	gpuErrchk(cudaUnbindTexture(texture_float_1));
	gpuErrchk(cudaUnbindTexture(texture_float2));

	inputVelocityBuffer = nextBufferIndex;
}

void simulateDensityAdvection()
{
	int nextBufferIndex = (inputDensityBuffer + 1) % 2;

	gpuErrchk(cudaBindTextureToArray(texture_float2, velocityBufferArray[inputVelocityBuffer], desc_float2));
	//texture_float2_in.filterMode = cudaFilterModeLinear;
	texture_float2.filterMode = cudaFilterModePoint;

	gpuErrchk(cudaBindTextureToArray(texture_float4, densityBufferArray[inputDensityBuffer], desc_float4));
	//texture_float2_in.filterMode = cudaFilterModeLinear;
	texture_float4.filterMode = cudaFilterModeLinear;

	gpuErrchk(cudaBindSurfaceToArray(surface_out_1, densityBufferArray[nextBufferIndex]));

	advectionDensity <<<numBlocks, threadsPerBlock >>> (gridResolution,
												  velocityBuffer[inputVelocityBuffer],
												  densityBuffer[inputDensityBuffer],
												  densityBuffer[nextBufferIndex]);
	gpuErrchk(cudaPeekAtLastError());
	gpuErrchk(cudaUnbindTexture(texture_float2));
	gpuErrchk(cudaUnbindTexture(texture_float4));

	inputDensityBuffer = nextBufferIndex;
}

void addForce(int x, int y, float2 force)
{
	float fx = (float)x / width;
	float fy = (float)y / height;

	gpuErrchk(cudaBindSurfaceToArray(surface_out_1, velocityBufferArray[inputVelocityBuffer]));
	gpuErrchk(cudaBindSurfaceToArray(surface_out_2, densityBufferArray[inputDensityBuffer]));

	addForceCUDA <<<numBlocks, threadsPerBlock >> > (fx, fy, force, gridResolution,
										  velocityBuffer[inputVelocityBuffer],
										  densityColor, densityBuffer[inputDensityBuffer]);
	gpuErrchk(cudaPeekAtLastError());
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
		gpuErrchk(cudaBindTextureToArray(texture_float4, densityBufferArray[inputDensityBuffer], desc_float4));
		texture_float4.filterMode = cudaFilterModePoint;

		visualizationDensity <<<numBlocks, threadsPerBlock >> > (width, height,
														  visualizationBufferGPU,
														  gridResolution,
														  densityBuffer[inputDensityBuffer]);
		gpuErrchk(cudaPeekAtLastError());
		gpuErrchk(cudaUnbindTexture(texture_float4));
		break;

	case 1:
		gpuErrchk(cudaBindTextureToArray(texture_float2, velocityBufferArray[inputVelocityBuffer], desc_float2));
		texture_float2.filterMode = cudaFilterModePoint;

		//TODO VisualizationGPU?
		visualizationVelocity <<<numBlocks, threadsPerBlock >> > (width, height,
														  visualizationBufferGPU,
														  gridResolution,
														  velocityBuffer[inputVelocityBuffer]);
		gpuErrchk(cudaPeekAtLastError());
		gpuErrchk(cudaUnbindTexture(texture_float2));
		break;

	case 2:
		gpuErrchk(cudaBindTextureToArray(texture_float_1, pressureBufferArray[inputPressureBuffer], desc_float));
		texture_float_1.filterMode = cudaFilterModePoint;


		visualizationPressure <<<numBlocks, threadsPerBlock >> > (width, height,
														  visualizationBufferGPU,
														  gridResolution,
														  pressureBuffer[inputPressureBuffer]);
		gpuErrchk(cudaPeekAtLastError());
		gpuErrchk(cudaUnbindTexture(texture_float_1));
		break;
	}

	gpuErrchk(cudaMemcpy(visualizationBufferCPU, visualizationBufferGPU, sizeof(float4) * width * height, cudaMemcpyDeviceToHost));

	glDrawPixels(width, height, GL_RGBA, GL_FLOAT, visualizationBufferCPU);
}

// OpenGL
int method = 1;
bool keysPressed[256];

void initOpenGL()
{
	glewExperimental = GL_TRUE;
	GLenum err = glewInit();
	if(GLEW_OK != err)
	{
		std::cerr << "Error: " << glewGetErrorString(err) << std::endl;
	}
	else
	{
		if(GLEW_VERSION_3_0)
		{
			std::cout << "Driver supports OpenGL 3.0\nDetails:" << std::endl;
			std::cout << "  Using GLEW " << glewGetString(GLEW_VERSION) << std::endl;
			std::cout << "  Vendor: " << glGetString(GL_VENDOR) << std::endl;
			std::cout << "  Renderer: " << glGetString(GL_RENDERER) << std::endl;
			std::cout << "  Version: " << glGetString(GL_VERSION) << std::endl;
			std::cout << "  GLSL: " << glGetString(GL_SHADING_LANGUAGE_VERSION) << std::endl;
		}
	}

	glClearColor(0.17f, 0.4f, 0.6f, 1.0f);
}

void display()
{
	//static int i = 0;
	//if(++i > 10)
	//	glutLeaveMainLoop();

	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
	glDisable(GL_DEPTH_TEST);

	simulationStep();
	visualizationStep();

	glEnable(GL_DEPTH_TEST);
	glutSwapBuffers();
}

void idle()
{
	glutPostRedisplay();
}

void keyDown(unsigned char key, int x, int y)
{
	keysPressed[key] = true;
}

void keyUp(unsigned char key, int x, int y)
{

	keysPressed[key] = false;
	switch(key)
	{
	case 'r':
		resetSimulation();
		break;

	case 'd':
		visualizationMethod = 0;
		break;
	case 'v':
		visualizationMethod = 1;
		break;
	case 'p':
		visualizationMethod = 2;
		break;

	case '1':
		densityColor.x = densityColor.y = densityColor.z = densityColor.w = 1.0f;
		break;

	case '2':
		densityColor.x = 1.0f;
		densityColor.y = densityColor.z = densityColor.w = 0.0f;
		break;

	case '3':
		densityColor.y = 1.0f;
		densityColor.x = densityColor.z = densityColor.w = 0.0f;
		break;

	case '4':
		densityColor.z = 1.0f;
		densityColor.x = densityColor.y = densityColor.w = 0.0f;
		break;

	case 27:
		glutLeaveMainLoop();
		break;
	}
}

int mX, mY;

void mouseClick(int button, int state, int x, int y)
{
	if(button == GLUT_LEFT_BUTTON)
		if(state == GLUT_DOWN)
		{
			mX = x;
			mY = y;
		}
}

void mouseMove(int x, int y)
{
	force.x = (float)(x - mX);
	force.y = -(float)(y - mY);
	//addForce(mX, height - mY, force);
	addForce(height / 2, width / 2, force);
	mX = x;
	mY = y;
}

void reshape(int newWidth, int newHeight)
{
	width = newWidth;
	height = newHeight;
	glViewport(0, 0, width, height);
}

int main(int argc, char* argv[])
{
	glutInit(&argc, argv);
	glutInitContextVersion(3, 0);
	glutInitContextFlags(GLUT_CORE_PROFILE | GLUT_DEBUG);
	glutInitDisplayMode(GLUT_RGBA | GLUT_DEPTH | GLUT_DOUBLE);
	glutInitWindowSize(width, height);
	glutCreateWindow("GPGPU: Incompressible fluid simulation");

	initOpenGL();

	glutDisplayFunc(display);
	glutIdleFunc(idle);
	glutReshapeFunc(reshape);
	glutKeyboardFunc(keyDown);
	glutKeyboardUpFunc(keyUp);
	glutMouseFunc(mouseClick);
	glutMotionFunc(mouseMove);

	// OpenCL processing
	initBuffers();

	glutMainLoop();

	gpuErrchk(cudaDeviceSynchronize());
	gpuErrchk(cudaFree(velocityBuffer[0]));
	gpuErrchk(cudaFree(velocityBuffer[1]));
	gpuErrchk(cudaFree(densityBuffer[0]));
	gpuErrchk(cudaFree(densityBuffer[1]));
	gpuErrchk(cudaFree(pressureBuffer[0]));
	gpuErrchk(cudaFree(pressureBuffer[1]));
	gpuErrchk(cudaFree(divergenceBuffer));
	gpuErrchk(cudaFree(vorticityBuffer));
	gpuErrchk(cudaFreeHost(visualizationBufferCPU));
	

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
