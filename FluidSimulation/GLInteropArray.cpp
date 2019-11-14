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
	glGenBuffers(1, &pbo);
	glBindBuffer(GL_PIXEL_UNPACK_BUFFER, pbo);
	unsigned int size = width * height * 4 * sizeof(float);
	glBufferData(GL_PIXEL_UNPACK_BUFFER, size, 0, GL_DYNAMIC_DRAW);
	glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);

	// register this buffer object with CUDA
	checkCudaErrors(cudaGraphicsGLRegisterBuffer(&cuda_vbo_resource, pbo, cudaGraphicsMapFlagsWriteDiscard));

	// Enable Texturing
	glEnable(GL_TEXTURE_2D);

	// Generate a texture ID
	glGenTextures(1, &texture);
	glBindTexture(GL_TEXTURE_2D, texture);

	glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA32F, width, height, 0, GL_RGBA, GL_FLOAT, NULL);

	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);

	// Make this the current texture (remember that GL is state-based)
}

GLInteropArray::~GLInteropArray()
{
	// unregister this buffer object with CUDA
	checkCudaErrors(cudaGraphicsUnregisterResource(cuda_vbo_resource));

	glBindBuffer(1, pbo);
	glDeleteBuffers(1, &pbo);

	pbo = 0;
}

GLuint GLInteropArray::getPBO()
{
	return pbo;
}

GLuint GLInteropArray::getTexture()
{
	glBindBuffer(GL_PIXEL_UNPACK_BUFFER, pbo);
	glBindTexture(GL_TEXTURE_2D, texture);
	glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, 4*96, 4 * 96, GL_RGBA, GL_FLOAT, NULL);
	return GLuint();
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
