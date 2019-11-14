#include "GLInteropArray.h"

#include <GL/freeglut.h>
#include <helper_gl.h>

#include <cuda_runtime.h>
#include <cuda_gl_interop.h>
#include <cuda.h>

#include <helper_functions.h>
#include <helper_cuda.h>

GLInteropArray::GLInteropArray(unsigned int width, unsigned int height)
{
	// create pixel buffer object
	glGenBuffers(1, &vbo);
	glBindBuffer(GL_PIXEL_UNPACK_BUFFER, vbo);
	unsigned int size = width * height * 4 * sizeof(float);
	glBufferData(GL_PIXEL_UNPACK_BUFFER, size, 0, GL_DYNAMIC_DRAW);
	glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);

	// register this buffer object with CUDA
	checkCudaErrors(cudaGraphicsGLRegisterBuffer(&cuda_vbo_resource, vbo, cudaGraphicsMapFlagsWriteDiscard));
}

GLInteropArray::~GLInteropArray()
{
	// unregister this buffer object with CUDA
	checkCudaErrors(cudaGraphicsUnregisterResource(cuda_vbo_resource));

	glBindBuffer(1, vbo);
	glDeleteBuffers(1, &vbo);

	vbo = 0;
}

GLuint GLInteropArray::getVBO()
{
	return vbo;
}

float4* GLInteropArray::getDataPointer()
{
	return dataPointer;
}

void GLInteropArray::map()
{
	checkCudaErrors(cudaGraphicsMapResources(1, &cuda_vbo_resource, 0));
	size_t num_bytes;
	checkCudaErrors(cudaGraphicsResourceGetMappedPointer((void**)&dataPointer, &num_bytes, cuda_vbo_resource));
}

void GLInteropArray::unmap()
{
	checkCudaErrors(cudaGraphicsUnmapResources(1, &cuda_vbo_resource, 0));
}
