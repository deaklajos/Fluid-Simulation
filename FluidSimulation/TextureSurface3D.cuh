#pragma once

#include <helper_cuda.h>
#include "cuda_runtime.h"
#include <cuda_runtime_api.h>

class TextureSurface3D
{
public:
	TextureSurface3D(const cudaChannelFormatDesc& descriptor, size_t size);
	~TextureSurface3D();

	cudaTextureObject_t getTexture() const;
	cudaSurfaceObject_t getSurface() const;
	cudaArray* getArray() const;

private:
	cudaArray* dataArray = nullptr;
	cudaTextureObject_t texture;
	cudaSurfaceObject_t surface;
};

