/*
 * TODO
 */


#pragma once


#include <windows.h>

// OpenGL
#include <GL\glew.h>


static const GLchar* vertexShaderSource[]
{
	"#version 450 core                          			\n"
	"                                           			\n"
	"layout (location = 0) in vec2 position;    			\n"
	"layout (location = 1) in vec3 color;       			\n"
	"                                           			\n"
	"out vec3 vs_color;                         			\n"
	"                                           			\n"
	"                                           			\n"
	"void main(void)                            			\n"
	"{                                          			\n"
	"    gl_Position = vec4(position.x, position.y, 0, 0);	\n"
	"}                                          			\n"
};

static const GLchar* fragmentShaderSource[]
{
	"#version 450 core                          			\n"
	"                                           			\n"
	"in vec3 vs_color;                          			\n"
	"                                           			\n"
	"out vec3 color;                            			\n"
	"                                           			\n"
	"void main(void)                            			\n"
	"{                                          			\n"
	"    color = vs_color;                      			\n"
	"}                                          			\n"
};