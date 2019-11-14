#pragma once

#include <GL/glew.h>

#include <vector_types.h>

class GLInteropArray
{
public:
	GLInteropArray(unsigned int width, unsigned int height);
	~GLInteropArray();
	GLuint getPBO();
	GLuint getTexture();
	float4* getDataPointer();
	void map();
	void unmap();

private:
	GLuint pbo;
	GLuint texture;
	float4* dataPointer = nullptr;
	struct cudaGraphicsResource* cuda_vbo_resource;
};

