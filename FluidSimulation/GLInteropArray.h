#pragma once

#include <GL/glew.h>

#include <vector_types.h>

class GLInteropArray
{
public:
	GLInteropArray(unsigned int width, unsigned int height);
	~GLInteropArray();
	GLuint getVBO();
	float4* getDataPointer();

private:
	GLuint vbo;
	float4* dataPointer;
	struct cudaGraphicsResource* cuda_vbo_resource;
};

