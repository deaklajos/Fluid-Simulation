#include "IntelliSenseErrorFixer.hpp"

#include "surface_indirect_functions.h"
#include "device_launch_parameters.h"
#include "cuda_runtime.h"
#include "cuda_runtime_api.h"

#include "helper_math.h"
#include <cuda.h>
#include <helper_cuda.h>
#include <helper_functions.h> 
#include <surface_functions.h> 

#include <stdio.h>
#include <iostream>

#include <GL/glew.h>
#include <GL/freeglut.h>

#include "IntelliSenseErrorFixer.hpp"

#include "TextureSurface3D.cuh"

// Kernels
cudaChannelFormatDesc desc_float = cudaCreateChannelDesc<float>();
cudaChannelFormatDesc desc_float2 = cudaCreateChannelDesc<float2>();
cudaChannelFormatDesc desc_float4 = cudaCreateChannelDesc<float4>();

// TODO make this into a define or constexpr
__constant__ float dt = 0.1f;
__device__ int cnt = 0;

#define gridResolution 96

__global__
void resetSimulationCUDA(
	cudaSurfaceObject_t velocityBuffer,
	cudaSurfaceObject_t pressureBuffer,
	cudaSurfaceObject_t densityBuffer)
{
	const unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
	const unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;
	const unsigned int z = blockIdx.z * blockDim.z + threadIdx.z;

	if (x < gridResolution && y < gridResolution && z < gridResolution)
	{
		surf3Dwrite(make_float4(0.0f), velocityBuffer, x * sizeof(float4), y, z);
		surf3Dwrite(0.0f, pressureBuffer, x * sizeof(float), y, z);
		surf3Dwrite(make_float4(0.0f), densityBuffer, x * sizeof(float4), y, z);
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
float2 getBil(float2 p, float2* buffer)
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
float4 getBil4(float2 p, float4* buffer)
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
void advection(cudaTextureObject_t inputVelocityBuffer, cudaSurfaceObject_t outputVelocityBuffer)
{
	const unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
	const unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;
	const unsigned int z = blockIdx.z * blockDim.z + threadIdx.z;

	if (x > 0 && x < gridResolution - 1 &&
		y > 0 && y < gridResolution - 1 &&
		z > 0 && z < gridResolution - 1)
	{
		const float4 velocity = tex3D<float4>(inputVelocityBuffer, x + 0.5f, y + 0.5f, z + 0.5f);
		float4 p = make_float4((float)x - dt * velocity.x, (float)y - dt * velocity.y, (float)z - dt * velocity.z, 0.0f);

		//TODO think: set bordertype
		//TODO get it and write it after
		p = clamp(p, make_float4(0.0f), make_float4(gridResolution));
		const float4 element = tex3D<float4>(inputVelocityBuffer, p.x + 0.5f, p.y + 0.5f, p.z + 0.5f);
		surf3Dwrite(element, outputVelocityBuffer, x * sizeof(float4), y, z);
	}
	else
	{
		if (x == 0)
		{
			// TODO Make Const
			float4 element = tex3D<float4>(inputVelocityBuffer, x + 1 + 0.5f, y + 0.5f, z + 0.5f);
			surf3Dwrite(-element, outputVelocityBuffer, x * sizeof(float4), y, z);
		}
		else if (x == gridResolution - 1)
		{
			float4 element = tex3D<float4>(inputVelocityBuffer, x - 1 + 0.5f, y + 0.5f, z + 0.5f);
			surf3Dwrite(-element, outputVelocityBuffer, x * sizeof(float4), y, z);
		}

		if (y == 0)
		{
			// TODO Make Const
			float4 element = tex3D<float4>(inputVelocityBuffer, x + 0.5f, y + 1 + 0.5f, z + 0.5f);
			surf3Dwrite(-element, outputVelocityBuffer, x * sizeof(float4), y, z);
		}
		else if (y == gridResolution - 1)
		{
			float4 element = tex3D<float4>(inputVelocityBuffer, x + 0.5f, y - 1 + 0.5f, z + 0.5f);
			surf3Dwrite(-element, outputVelocityBuffer, x * sizeof(float4), y, z);
		}

		if (z == 0)
		{
			// TODO Make Const
			float4 element = tex3D<float4>(inputVelocityBuffer, x + 0.5f, y+ 0.5f, z + 1 + 0.5f);
			surf3Dwrite(-element, outputVelocityBuffer, x * sizeof(float4), y, z);
		}
		else if (z == gridResolution - 1)
		{
			float4 element = tex3D<float4>(inputVelocityBuffer, x + 0.5f, y+ 0.5f, z - 1 + 0.5f);
			surf3Dwrite(-element, outputVelocityBuffer, x * sizeof(float4), y, z);
		}
	}
}

__global__
void advectionDensity(cudaTextureObject_t velocityBuffer, cudaTextureObject_t inputDensityBuffer, cudaSurfaceObject_t outputDensityBuffer)
{
	const unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
	const unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;
	const unsigned int z = blockIdx.z * blockDim.z + threadIdx.z;

	if (x > 0 && x < gridResolution - 1 &&
		y > 0 && y < gridResolution - 1 &&
		z > 0 && z < gridResolution - 1)
	{
		const float4 velocity = tex3D<float4>(velocityBuffer, x + 0.5f, y + 0.5f, z + 0.5f);

		float4 p = float4{ (float)x - dt * velocity.x, (float)y - dt * velocity.y, (float)z - dt * velocity.z, 0.0f};

		p = clamp(p, make_float4(0.0f), make_float4(gridResolution));
		const float4 element = tex3D<float4>(inputDensityBuffer, p.x + 0.5f, p.y + 0.5f, p.z + 0.5f);
		surf3Dwrite(element, outputDensityBuffer, x * sizeof(float4), y, z);
	}
	else
	{
		surf3Dwrite(make_float4(0.0f), outputDensityBuffer, x * sizeof(float4), y, z);
	}
}

//TODO remove debug function
__global__ void myprint()
{
	printf("[%d, %d]\n", blockIdx.y * gridDim.x + blockIdx.x, blockIdx.y * gridDim.y + blockIdx.y);
}

__global__
void diffusion(cudaTextureObject_t inputVelocityBuffer, cudaSurfaceObject_t outputVelocityBuffer)
{
	const unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
	const unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;
	const unsigned int z = blockIdx.z * blockDim.z + threadIdx.z;

	const float viscousity = 0.01f;
	const float alpha = 1.0f / (viscousity * dt);
	const float beta = 1.0f / (6.0f + alpha);

	if (x > 0 && x < gridResolution - 1 &&
		y > 0 && y < gridResolution - 1 &&
		z > 0 && z < gridResolution - 1)
	{
		const float4 vL = tex3D<float4>(inputVelocityBuffer, x - 1 + 0.5f, y + 0.5f, z + 0.5f);
		const float4 vR = tex3D<float4>(inputVelocityBuffer, x + 1 + 0.5f, y + 0.5f, z + 0.5f);
		const float4 vB = tex3D<float4>(inputVelocityBuffer, x + 0.5f, y - 1 + 0.5f, z + 0.5f);
		const float4 vT = tex3D<float4>(inputVelocityBuffer, x + 0.5f, y + 1 + 0.5f, z + 0.5f);
		const float4 vN = tex3D<float4>(inputVelocityBuffer, x + 0.5f, y + 0.5f, z - 1 + 0.5f);
		const float4 vF = tex3D<float4>(inputVelocityBuffer, x + 0.5f, y + 0.5f, z + 1 + 0.5f);

		const float4 velocity = tex3D<float4>(inputVelocityBuffer, x + 0.5f, y + 0.5f, z + 0.5f);

		const float4 out = (vL + vR + vB + vT + vN + vF + alpha * velocity) * beta;

		surf3Dwrite<float4>(out, outputVelocityBuffer, x * sizeof(float4), y, z);
	}
	else
	{
		const float4 velocity = tex3D<float4>(inputVelocityBuffer, x + 0.5f, y + 0.5f, z + 0.5f);

		surf3Dwrite<float4>(velocity, outputVelocityBuffer, x * sizeof(float4), y, z);
	}
}

//__global__
//void vorticity(float2* velocityBuffer, float* vorticityBuffer)
//{
//	uint2 id{ blockIdx.x * blockDim.x + threadIdx.x, blockIdx.y * blockDim.y + threadIdx.y };
//
//	if (id.x > 0 && id.x < gridResolution - 1 &&
//		id.y > 0 && id.y < gridResolution - 1)
//	{
//		float2 vL = tex2D(texture_float2, id.x - 1 + 0.5f, id.y + 0.5f);
//		float2 vR = tex2D(texture_float2, id.x + 1 + 0.5f, id.y + 0.5f);
//		float2 vB = tex2D(texture_float2, id.x + 0.5f, id.y - 1 + 0.5f);
//		float2 vT = tex2D(texture_float2, id.x + 0.5f, id.y + 1 + 0.5f);
//
//		float out = (vR.y - vL.y) - (vT.x - vB.x);
//
//		surf2Dwrite(out, surface_out_1, id.x * sizeof(float), id.y);
//	}
//	else
//	{
//		surf2Dwrite(0.0f, surface_out_1, id.x * sizeof(float), id.y);
//	}
//}

//__global__
//void addVorticity(float* vorticityBuffer, float2* velocityBuffer)
//{
//	uint2 id{ blockIdx.x * blockDim.x + threadIdx.x, blockIdx.y * blockDim.y + threadIdx.y };
//
//	const float scale = 0.2f;
//
//	if (id.x > 0 && id.x < gridResolution - 1 &&
//		id.y > 0 && id.y < gridResolution - 1)
//	{
//		float vL = tex2D(texture_float_1, id.x - 1 + 0.5f, id.y + 0.5f);
//		float vR = tex2D(texture_float_1, id.x + 1 + 0.5f, id.y + 0.5f);
//		float vB = tex2D(texture_float_1, id.x + 0.5f, id.y - 1 + 0.5f);
//		float vT = tex2D(texture_float_1, id.x + 0.5f, id.y + 1 + 0.5f);
//
//		float4 gradV{ vR - vL, vT - vB, 0.0f, 0.0f };
//		float4 z{ 0.0f, 0.0f, 1.0f, 0.0f };
//
//		if (dot(gradV, gradV))
//		{
//			float4 vorticityForce = make_float4(scale * cross(make_float3(gradV), make_float3(z)));
//
//			float2 temp;
//			surf2Dread(&temp, surface_out_1, id.x * sizeof(float2), id.y);
//
//			temp += make_float2(vorticityForce.x, vorticityForce.y) * dt;
//			surf2Dwrite(temp, surface_out_1, id.x * sizeof(float2), id.y);
//		}
//	}
//}

__global__
void divergence(cudaTextureObject_t velocityBuffer, cudaSurfaceObject_t divergenceBuffer)
{
	const unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
	const unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;
	const unsigned int z = blockIdx.z * blockDim.z + threadIdx.z;

	if (x > 0 && x < gridResolution - 1 &&
		y > 0 && y < gridResolution - 1 &&
		z > 0 && z < gridResolution - 1)
	{
		const float4 vL = tex3D<float4>(velocityBuffer, x - 1 + 0.5f, y + 0.5f, z + 0.5f);
		const float4 vR = tex3D<float4>(velocityBuffer, x + 1 + 0.5f, y + 0.5f, z + 0.5f);
		const float4 vB = tex3D<float4>(velocityBuffer, x + 0.5f, y - 1 + 0.5f, z + 0.5f);
		const float4 vT = tex3D<float4>(velocityBuffer, x + 0.5f, y + 1 + 0.5f, z + 0.5f);
		const float4 vN = tex3D<float4>(velocityBuffer, x + 0.5f, y + 0.5f, z - 1 + 0.5f);
		const float4 vF = tex3D<float4>(velocityBuffer, x + 0.5f, y + 0.5f, z + 1 + 0.5f);

		const float out = (1 / 3.0f) * ((vR.x - vL.x) + (vT.y - vB.y) + (vF.z - vN.z));
		surf3Dwrite<float>(out, divergenceBuffer, x * sizeof(float), y, z);
	}
	else
	{
		surf3Dwrite<float>(0.0f, divergenceBuffer, x * sizeof(float), y, z);
	}
}

__global__
void pressureJacobi(cudaTextureObject_t inputPressureBuffer, cudaSurfaceObject_t outputPressureBuffer, cudaTextureObject_t divergenceBuffer)
{
	const unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
	const unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;
	const unsigned int z = blockIdx.z * blockDim.z + threadIdx.z;

	if (x > 0 && x < gridResolution - 1 &&
		y > 0 && y < gridResolution - 1 &&
		z > 0 && z < gridResolution - 1)
	{
		const float alpha = -1.0f;
		const float beta = 1 / 6.0f;

		// TODO this access redundant
		const float pL = tex3D<float>(inputPressureBuffer, x - 1 + 0.5f, y + 0.5f, z + 0.5f);
		const float pR = tex3D<float>(inputPressureBuffer, x + 1 + 0.5f, y + 0.5f, z + 0.5f);
		const float pB = tex3D<float>(inputPressureBuffer, x + 0.5f, y - 1 + 0.5f, z + 0.5f);
		const float pT = tex3D<float>(inputPressureBuffer, x + 0.5f, y + 1 + 0.5f, z + 0.5f);
		const float pN = tex3D<float>(inputPressureBuffer, x + 0.5f, y + 0.5f, z - 1 + 0.5f);
		const float pF = tex3D<float>(inputPressureBuffer, x + 0.5f, y + 0.5f, z + 1 + 0.5f);

		const float divergence = tex3D<float>(divergenceBuffer, x + 0.5f, y + 0.5f, z + 0.5f);

		const float out = (pL + pR + pB + pT + pN + pF + alpha * divergence) * beta;

		surf3Dwrite(out, outputPressureBuffer, x * sizeof(float), y, z);
	}
	else
	{
		if (x == 0)
		{
			const float element = tex3D<float>(inputPressureBuffer, x + 1 + 0.5f, y + 0.5f, z + 0.5f);
			surf3Dwrite(-element, outputPressureBuffer, x * sizeof(float), y, z);
		}
		else if (x == gridResolution - 1)
		{
			const float element = tex3D<float>(inputPressureBuffer, x - 1 + 0.5f, y + 0.5f, z + 0.5f);
			surf3Dwrite(-element, outputPressureBuffer, x * sizeof(float), y, z);
		}

		if (y == 0)
		{
			const float element = tex3D<float>(inputPressureBuffer, x + 0.5f, y + 1 + 0.5f, z + 0.5f);
			surf3Dwrite(-element, outputPressureBuffer, x * sizeof(float), y, z);
		}
		else if (y == gridResolution - 1)
		{
			const float element = tex3D<float>(inputPressureBuffer, x + 0.5f, y - 1 + 0.5f, z + 0.5f);
			surf3Dwrite(-element, outputPressureBuffer, x * sizeof(float), y, z);
		}

		if (z == 0)
		{
			const float element = tex3D<float>(inputPressureBuffer, x + 0.5f, y + 0.5f, z + 1 + 0.5f);
			surf3Dwrite(-element, outputPressureBuffer, x * sizeof(float), y, z);
		}
		else if (z == gridResolution - 1)
		{
			const float element = tex3D<float>(inputPressureBuffer, x + 0.5f, y + 0.5f, z - 1 + 0.5f);
			surf3Dwrite(-element, outputPressureBuffer, x * sizeof(float), y, z);
		}
	}
}

__global__
void projectionCUDA(cudaTextureObject_t inputVelocityBuffer, cudaTextureObject_t pressureBuffer, cudaSurfaceObject_t outputVelocityBuffer)
{
	const unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
	const unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;
	const unsigned int z = blockIdx.z * blockDim.z + threadIdx.z;

	if (x > 0 && x < gridResolution - 1 &&
		y > 0 && y < gridResolution - 1 &&
		z > 0 && z < gridResolution - 1)
	{
		// TODO this access redundant
		const float pL = tex3D<float>(pressureBuffer, x - 1 + 0.5f, y + 0.5f, z + 0.5f);
		const float pR = tex3D<float>(pressureBuffer, x + 1 + 0.5f, y + 0.5f, z + 0.5f);
		const float pB = tex3D<float>(pressureBuffer, x + 0.5f, y - 1 + 0.5f, z + 0.5f);
		const float pT = tex3D<float>(pressureBuffer, x + 0.5f, y + 1 + 0.5f, z + 0.5f);
		const float pN = tex3D<float>(pressureBuffer, x + 0.5f, y + 0.5f, z - 1 + 0.5f);
		const float pF = tex3D<float>(pressureBuffer, x + 0.5f, y + 0.5f, z + 1 + 0.5f);

		float4 velocity = tex3D<float4>(inputVelocityBuffer, x + 0.5f, y + 0.5f, z + 0.5f);
		float4 out = velocity - float4{ pR - pL, pT - pB, pF - pN, 0.0f };

		surf3Dwrite(out, outputVelocityBuffer, x * sizeof(float4), y, z);
	}
	else
	{
		if (x == 0)
		{
			// TODO Make Const
			float4 element = tex3D<float4>(inputVelocityBuffer, x + 1 + 0.5f, y + 0.5f, z + 0.5f);
			surf3Dwrite(-element, outputVelocityBuffer, x * sizeof(float4), y, z);
		}
		else if (x == gridResolution - 1)
		{
			float4 element = tex3D<float4>(inputVelocityBuffer, x - 1 + 0.5f, y + 0.5f, z + 0.5f);
			surf3Dwrite(-element, outputVelocityBuffer, x * sizeof(float4), y, z);
		}

		if (y == 0)
		{
			// TODO Make Const
			float4 element = tex3D<float4>(inputVelocityBuffer, x + 0.5f, y + 1 + 0.5f, z + 0.5f);
			surf3Dwrite(-element, outputVelocityBuffer, x * sizeof(float4), y, z);
		}
		else if (y == gridResolution - 1)
		{
			float4 element = tex3D<float4>(inputVelocityBuffer, x + 0.5f, y - 1 + 0.5f, z + 0.5f);
			surf3Dwrite(-element, outputVelocityBuffer, x * sizeof(float4), y, z);
		}

		if (z == 0)
		{
			// TODO Make Const
			float4 element = tex3D<float4>(inputVelocityBuffer, x + 0.5f, y + 0.5f, z + 1 + 0.5f);
			surf3Dwrite(-element, outputVelocityBuffer, x * sizeof(float4), y, z);
		}
		else if (z == gridResolution - 1)
		{
			float4 element = tex3D<float4>(inputVelocityBuffer, x + 0.5f, y + 0.5f, z - 1 + 0.5f);
			surf3Dwrite(-element, outputVelocityBuffer, x * sizeof(float4), y, z);
		}
	}
}

// TODO Do this in linear?
__global__
void addForceCUDA(float xIndex, float yIndex, float zIndex, const float4 force,
	cudaSurfaceObject_t velocityBuffer, const float4 density, cudaSurfaceObject_t densityBuffer)
{
	const unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
	const unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;
	const unsigned int z = blockIdx.z * blockDim.z + threadIdx.z;

	const float dx = ((float)x / (float)gridResolution) - xIndex;
	const float dy = ((float)y / (float)gridResolution) - yIndex;
	const float dz = ((float)z / (float)gridResolution) - zIndex;

	const float radius = 0.001f;

	const float c = exp(-(dx * dx + dy * dy + dz * dz) / radius) * dt;

	// TODO remove variable
	//cnt = 0;

	const float4 outVelocity = surf3Dread<float4>(velocityBuffer, x * (int)sizeof(float4), y, z);
	surf3Dwrite(outVelocity + c * force, velocityBuffer, x * (int)sizeof(float4), y, z);

	float4 outDensity = surf3Dread<float4>(densityBuffer, x * sizeof(float4), y, z);
	surf3Dwrite(outDensity + c * density, densityBuffer, x * sizeof(float4), y, z);
}

// *************
// Visualization
// *************

__global__
void visualizationDensity(float4* visualizationBuffer, cudaTextureObject_t densityBuffer)
{
	const unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
	const unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;
	const unsigned int z = blockIdx.z * blockDim.z + threadIdx.z;

	if (x < gridResolution && y < gridResolution && z == gridResolution / 2)
	{
		const float4 density = tex3D<float4>(densityBuffer, x + 0.5f, y + 0.5f, z + 0.5f);
		visualizationBuffer[x + y * gridResolution] = density;
	}
}

// TODO 3D
__global__
void visualizationVelocity(float4* visualizationBuffer, cudaTextureObject_t velocityBuffer)
{
	const unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
	const unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;
	const unsigned int z = blockIdx.z * blockDim.z + threadIdx.z;

	if (x < gridResolution && y < gridResolution && z == gridResolution / 2)
	{
		const float4 velocity = tex3D<float4>(velocityBuffer, x + 0.5f, y + 0.5f, z + 0.5f);
		//const float4 velocity = surf3Dread<float4>(velocityBuffer, x * (int)sizeof(float4), y, z);

		const float4 tmp = (1.0f + velocity) / 2.0f;
		visualizationBuffer[x + y * gridResolution] = float4{ tmp.x, tmp.y, tmp.z, 0.0f };
	}
}

__global__
void visualizationPressure(float4* visualizationBuffer, cudaTextureObject_t pressureBuffer)
{
	const unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
	const unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;
	const unsigned int z = blockIdx.z * blockDim.z + threadIdx.z;

	if (x < gridResolution && y < gridResolution && z == gridResolution / 2)
	{
		const float pressure = tex3D<float>(pressureBuffer, x + 0.5f, y + 0.5f, z + 0.5f);

		visualizationBuffer[x + y * gridResolution] = make_float4((1.0f + pressure) / 2.0f);
	}
}

// End of kernels

// Buffers

// simulation
//int gridResolution = 192;
dim3 threadsPerBlock(8, 8, 8);
dim3 numBlocks(gridResolution / threadsPerBlock.x, gridResolution / threadsPerBlock.y, gridResolution / threadsPerBlock.z);

int inputVelocityBuffer = 0;
TextureSurface3D* velocityBuffer[2];

int inputDensityBuffer = 0;
TextureSurface3D* densityBuffer[2];
float4 densityColor;

int inputPressureBuffer = 0;
TextureSurface3D* pressureBuffer[2];

TextureSurface3D* divergenceBuffer;
float* vorticityBuffer;

float3 force;

// visualization
int width = gridResolution;
int height = gridResolution;

float4* visualizationBufferGPU;
float4* visualizationBufferCPU;

int visualizationMethod = 1;

// End of Buffers
void addForce(int x, int y, float3 force);
void resetSimulation();

void initBuffers()
{
	velocityBuffer[0] = new TextureSurface3D(desc_float4, gridResolution);
	velocityBuffer[1] = new TextureSurface3D(desc_float4, gridResolution);

	densityBuffer[0] = new TextureSurface3D(desc_float4, gridResolution);
	densityBuffer[1] = new TextureSurface3D(desc_float4, gridResolution);

	pressureBuffer[0] = new TextureSurface3D(desc_float, gridResolution);
	pressureBuffer[1] = new TextureSurface3D(desc_float, gridResolution);

	divergenceBuffer = new TextureSurface3D(desc_float, gridResolution);

	checkCudaErrors(cudaMalloc(&vorticityBuffer, sizeof(float) * gridResolution * gridResolution));

	densityColor = float4{ 1.0f, 1.0f, 1.0f, 1.0f };

	checkCudaErrors(cudaMallocHost(&visualizationBufferCPU, sizeof(float4) * width * height));
	checkCudaErrors(cudaMalloc(&visualizationBufferGPU, sizeof(float4) * width * height));

	resetSimulation();
}

void resetSimulation()
{
	resetSimulationCUDA KERNEL_CALL(numBlocks, threadsPerBlock)(
		velocityBuffer[inputVelocityBuffer]->getSurface(),
		pressureBuffer[inputPressureBuffer]->getSurface(),
		densityBuffer[inputDensityBuffer]->getSurface());

	checkCudaErrors(cudaPeekAtLastError());
}

void resetPressure()
{
	// TODO This could be made better
	resetSimulationCUDA KERNEL_CALL(numBlocks, threadsPerBlock)(
		velocityBuffer[(inputVelocityBuffer + 1) % 2]->getSurface(),
		pressureBuffer[inputPressureBuffer]->getSurface(),
		densityBuffer[(inputDensityBuffer + 1) % 2]->getSurface());

	checkCudaErrors(cudaPeekAtLastError());
}


void simulateAdvection()
{
	int nextBufferIndex = (inputVelocityBuffer + 1) % 2;

	advection KERNEL_CALL(numBlocks, threadsPerBlock)(
		velocityBuffer[inputVelocityBuffer]->getTexture(),
		velocityBuffer[nextBufferIndex]->getSurface());
	checkCudaErrors(cudaPeekAtLastError());

	inputVelocityBuffer = nextBufferIndex;
}

void simulateVorticity()
{
	//vorticity KERNEL_CALL(numBlocks, threadsPerBlock)(velocityBuffer[inputVelocityBuffer], vorticityBuffer);
	//checkCudaErrors(cudaPeekAtLastError());

	//addVorticity KERNEL_CALL(numBlocks, threadsPerBlock)(
	//	vorticityBuffer,
	//	velocityBuffer[inputVelocityBuffer]);
	//checkCudaErrors(cudaPeekAtLastError());
}

void simulateDiffusion()
{
	for (int i = 0; i < 10; ++i)
	{
		int nextBufferIndex = (inputVelocityBuffer + 1) % 2;

		diffusion KERNEL_CALL(numBlocks, threadsPerBlock)(
			velocityBuffer[inputVelocityBuffer]->getTexture(),
			velocityBuffer[nextBufferIndex]->getSurface());

		checkCudaErrors(cudaPeekAtLastError());

		inputVelocityBuffer = nextBufferIndex;
	}
}

void projection()
{
	divergence KERNEL_CALL(numBlocks, threadsPerBlock)(
		velocityBuffer[inputVelocityBuffer]->getTexture(),
		divergenceBuffer->getSurface());
	checkCudaErrors(cudaPeekAtLastError());

	resetPressure();

	// TODO Could This be done on the kernel level?
	for (int i = 0; i < 10; ++i)
	{
		int nextBufferIndex = (inputPressureBuffer + 1) % 2;

		pressureJacobi KERNEL_CALL(numBlocks, threadsPerBlock)(
			pressureBuffer[inputPressureBuffer]->getTexture(),
			pressureBuffer[nextBufferIndex]->getSurface(),
			divergenceBuffer->getTexture());

		checkCudaErrors(cudaPeekAtLastError());

		inputPressureBuffer = nextBufferIndex;
	}

	int nextBufferIndex = (inputVelocityBuffer + 1) % 2;

	projectionCUDA KERNEL_CALL(numBlocks, threadsPerBlock)(
		velocityBuffer[inputVelocityBuffer]->getTexture(),
		pressureBuffer[inputPressureBuffer]->getTexture(),
		velocityBuffer[nextBufferIndex]->getSurface());

	checkCudaErrors(cudaPeekAtLastError());

	inputVelocityBuffer = nextBufferIndex;
}

void simulateDensityAdvection()
{
	int nextBufferIndex = (inputDensityBuffer + 1) % 2;

	advectionDensity KERNEL_CALL(numBlocks, threadsPerBlock)(
		velocityBuffer[inputVelocityBuffer]->getTexture(),
		densityBuffer[inputDensityBuffer]->getTexture(),
		densityBuffer[nextBufferIndex]->getSurface());

	checkCudaErrors(cudaPeekAtLastError());

	inputDensityBuffer = nextBufferIndex;
}

void addForce(int x, int y, float3 force)
{
	// TODO Give meaning to this
	float fx = (float)x / width;
	float fy = (float)y / height;
	float fz = (float)y / height;

	float4 force4{ force.x, force.y, force.z, 0.0 };

	addForceCUDA KERNEL_CALL(numBlocks, threadsPerBlock)(
		fx, fy, fz, force4,
		velocityBuffer[inputVelocityBuffer]->getSurface(),
		densityColor, densityBuffer[inputDensityBuffer]->getSurface());
	checkCudaErrors(cudaPeekAtLastError());
}

void simulationStep()
{
	simulateAdvection();
	simulateDiffusion();
	//simulateVorticity();
	projection();
	simulateDensityAdvection();
}

void visualizationStep()
{
	switch (visualizationMethod)
	{
	case 0:
		visualizationDensity KERNEL_CALL(numBlocks, threadsPerBlock)(
			visualizationBufferGPU,
			densityBuffer[inputDensityBuffer]->getTexture());

		checkCudaErrors(cudaPeekAtLastError());
		break;

	case 1:
		visualizationVelocity KERNEL_CALL(numBlocks, threadsPerBlock)(
			visualizationBufferGPU,
			velocityBuffer[inputVelocityBuffer]->getTexture());

		checkCudaErrors(cudaPeekAtLastError());
		break;

	case 2:
		visualizationPressure KERNEL_CALL(numBlocks, threadsPerBlock)(
			visualizationBufferGPU,
			pressureBuffer[inputPressureBuffer]->getTexture());

		checkCudaErrors(cudaPeekAtLastError());
		break;
	}

	checkCudaErrors(cudaMemcpy(visualizationBufferCPU, visualizationBufferGPU, sizeof(float4) * width * height, cudaMemcpyDeviceToHost));

	glDrawPixels(width, height, GL_RGBA, GL_FLOAT, visualizationBufferCPU);
}

// OpenGL
int method = 1;
bool keysPressed[256];

void initOpenGL()
{
	glewExperimental = GL_TRUE;
	GLenum err = glewInit();
	if (GLEW_OK != err)
	{
		std::cerr << "Error: " << glewGetErrorString(err) << std::endl;
	}
	else
	{
		if (GLEW_VERSION_3_0)
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
	/*static int i = 0;
	if(++i > 100)
		glutLeaveMainLoop();*/

	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
	glDisable(GL_DEPTH_TEST);

	addForce(gridResolution / 2, gridResolution / 2, make_float3(1, 1, 0));
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
	switch (key)
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
	if (button == GLUT_LEFT_BUTTON)
		if (state == GLUT_DOWN)
		{
			mX = x;
			mY = y;
		}
}

void mouseMove(int x, int y)
{
	force.x = (float)(x - mX);
	force.y = -(float)(y - mY);
	force.z = 0.0f;
	//addForce(mX, height - mY, force); //old
	addForce(height / 2, width / 2, force);
	mX = x;
	mY = y;
}

void reshape(int newWidth, int newHeight)
{
	/*width = newWidth;
	height = newHeight;
	glViewport(0, 0, width, height);*/
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

	// Allows deallocations.
	glutSetOption(GLUT_ACTION_ON_WINDOW_CLOSE, GLUT_ACTION_CONTINUE_EXECUTION);

	// OpenCL processing
	initBuffers();

	glutMainLoop();

	checkCudaErrors(cudaDeviceSynchronize());

	delete velocityBuffer[0];
	delete velocityBuffer[1];

	delete densityBuffer[0];
	delete densityBuffer[1];

	delete pressureBuffer[0];
	delete pressureBuffer[1];

	delete divergenceBuffer;
	checkCudaErrors(cudaFree(vorticityBuffer));
	checkCudaErrors(cudaFree(visualizationBufferGPU));
	checkCudaErrors(cudaFreeHost(visualizationBufferCPU));

	// cudaDeviceReset must be called before exiting in order for profiling and
	// tracing tools such as Nsight and Visual Profiler to show complete traces.
	cudaError_t cudaStatus = cudaDeviceReset();
	if (cudaStatus != cudaSuccess)
	{
		fprintf(stderr, "cudaDeviceReset failed!");
		return 1;
	}

	return 0;
}
