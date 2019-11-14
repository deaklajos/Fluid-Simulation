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
	//assert(&vbo);

	// create buffer object
	glGenBuffers(1, &vbo);
	glBindBuffer(GL_ARRAY_BUFFER, vbo);

	// initialize buffer object
	unsigned int size = width * height * 4 * sizeof(float);
	glBufferData(GL_ARRAY_BUFFER, size, 0, GL_DYNAMIC_DRAW);

	glBindBuffer(GL_ARRAY_BUFFER, 0);

	// register this buffer object with CUDA
	checkCudaErrors(cudaGraphicsGLRegisterBuffer(&cuda_vbo_resource, vbo, cudaGraphicsMapFlagsWriteDiscard));

	SDK_CHECK_ERROR_GL();

	// map OpenGL buffer object for writing from CUDA
	checkCudaErrors(cudaGraphicsMapResources(1, &cuda_vbo_resource, 0));
	size_t num_bytes;
	checkCudaErrors(cudaGraphicsResourceGetMappedPointer((void**)&dataPointer, &num_bytes,
		cuda_vbo_resource));
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
