#include "TextureSurface3D.cuh"

TextureSurface3D::TextureSurface3D(const cudaChannelFormatDesc& chanelDescriptor, size_t size)
{
	checkCudaErrors(cudaMalloc3DArray(&dataArray, &chanelDescriptor, make_cudaExtent(size, size, size), cudaArraySurfaceLoadStore));

	cudaResourceDesc resourceDescriptor;
	memset(&resourceDescriptor, 0, sizeof(cudaResourceDesc));
	resourceDescriptor.resType = cudaResourceTypeArray;
	resourceDescriptor.res.array.array = dataArray;

	cudaTextureDesc textureDescriptor;
	memset(&textureDescriptor, 0, sizeof(cudaTextureDesc));
	textureDescriptor.normalizedCoords = false; // access with unnormalized texture coordinates
	textureDescriptor.filterMode = cudaFilterModeLinear; // linear interpolation
	// wrap texture coordinates
	textureDescriptor.addressMode[0] = cudaAddressModeWrap;
	textureDescriptor.addressMode[1] = cudaAddressModeWrap;
	textureDescriptor.addressMode[2] = cudaAddressModeWrap;
	textureDescriptor.readMode = cudaReadModeElementType;

	checkCudaErrors(cudaCreateTextureObject(&texture, &resourceDescriptor, &textureDescriptor, NULL));

	checkCudaErrors(cudaCreateSurfaceObject(&surface, &resourceDescriptor));
}

TextureSurface3D::~TextureSurface3D()
{
	checkCudaErrors(cudaDestroyTextureObject(texture));
	checkCudaErrors(cudaDestroySurfaceObject(surface));
	checkCudaErrors(cudaFreeArray(dataArray));
}

cudaTextureObject_t TextureSurface3D::getTexture() const
{
	return texture;
}

cudaSurfaceObject_t TextureSurface3D::getSurface() const
{
	return surface;
}

cudaArray* TextureSurface3D::getArray() const
{
	return dataArray;
}
