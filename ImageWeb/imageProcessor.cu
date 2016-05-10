/*
* TODO
*/


#include <float.h>
#include <math.h>
#ifdef _WIN32
#include <windows.h>
#endif

// OpenGL
#include <GL/glew.h>
#include <GL/GL.h>
#include <GL/glut.h>

// CUDA
#include <cuda_runtime.h>
#include <cuda_gl_interop.h>
#include <curand_kernel.h>
#include "scan.cu"

#include "constants.hpp"
#include "imageProcessor.hpp"
#include "util.hpp"


// Forward declarations
ReturnCode generateCDF(float* distribution, float* distrRows, float* distrCols,
	float* dev_distrRowSums, short2 dims, dim3 blockSize, dim3 threadSize);

__global__ void cuRAND_init(curandState_t* states, uint64_t seed);
__global__ void computePixelIntensities(Image img, ChannelBuf intensities, short2 dims);
__global__ void detectEdges(ChannelBuf intensities, ChannelBuf edges, short2 dims);
__global__ void blendDistributionData(ChannelBuf intensity, ChannelBuf edges,
	ChannelBuf historicity, float* distribution, short2 dims, float intEdgeWeight,
	float historicityWeight);
__global__ void copyRowSums(float* distrRows, float* distrRowSums, short2 dims);
__global__ void normalizeCDF(float* distrRows, float* distrRowSums, float* distrCols, short2 dims);
__global__ void samplePoints(float* rowDist, float* colDist, short2 dims,
	curandState_t* randStates, Point* pointBuf, uint32_t numPoints, ChannelBuf historicityBuf);
__global__ void computeVoronoi(uint32_t* voronoi, short2 dims, Point* points, uint32_t numPoints);
__global__ void convertToGLCoords(Point* points, short2 dims);
__global__ void constructTriangulation(uint32_t* voronoi, short2 dims, Point* points,
	Triangle* triangulation, uint32_t* numTriangles);
__global__ void assignTriangleColorings(Triangle* triangles, uint32_t* numTriangles, Image srcImg,
	ColoringMode mode, Color bgColor, Color fgColor, short2 dims);

__device__ uint16_t binarySearch(float* arr, size_t arrSize, uint32_t minIndex, uint32_t maxIndex,
	float target);


// Shader code
static const GLchar* vertexShaderSource[]
{
	"#version 450 core                                      \n"
		"                                                       \n"
		"layout (location = 0) in vec2 position;                \n"
		"layout (location = 1) in vec3 color;                   \n"
		"                                                       \n"
		"out vec3 vs_color;                                     \n"
		"                                                       \n"
		"void main(void)                                        \n"
		"{                                                      \n"
		"    gl_Position = vec4(position.x, position.y, 0, 1);  \n"
		"    vs_color = color;                                  \n"
		"}                                                      \n"
};

static const GLchar* fragmentShaderSource[]
{
	"#version 450 core                                      \n"
		"                                                       \n"
		"in vec3 vs_color;                                      \n"
		"                                                       \n"
		"out vec3 color;                                        \n"
		"                                                       \n"
		"void main(void)                                        \n"
		"{                                                      \n"
		"    color = vs_color;                                  \n"
		"}                                                      \n"
};


ReturnCode processImageResource(ImageResource& input, ImageResource& output, ParamBundle params)
{
	ReturnCode retCode;
	cudaError_t cudaRetCode;
	GLenum glRetCode = GL_NO_ERROR;
	int cudaDeviceHandle;

	// Prepare relevant state
	auto dims = make_short2(input.getWidth(), input.getHeight());
	uint32_t imgBufSize = dims.x * dims.y;
	uint32_t numPoints = static_cast<uint32_t>(imgBufSize * params.pointRatio);
	dim3 blockSize(dims.x / THREADS_PER_BLOCK, dims.y / THREADS_PER_BLOCK);
	auto pointDim = static_cast<uint32_t>(sqrt(numPoints / THREADS_PER_BLOCK / THREADS_PER_BLOCK));
	dim3 blockSizePoints(pointDim, pointDim);
	dim3 threadSize(THREADS_PER_BLOCK, THREADS_PER_BLOCK);

	// Ensure that a CUDA device with compute level 2.0 or greater is available
	cudaDeviceProp prop;
	prop.major = 2;
	prop.minor = 0;
	CUDA_CALL(cudaChooseDevice(&cudaDeviceHandle, &prop));
	CUDA_CALL(cudaSetDevice(cudaDeviceHandle));

	// Set up OpenGL with CUDA interoperability
	int argc = 1;
	char* argv = "";
	glutInit(&argc, &argv);
	glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGBA);
	glutInitWindowSize(1, 1);
	auto glutWindow = glutCreateWindow("Context Window");
	glutHideWindow();
	glewInit();
	CUDA_CALL(cudaGLSetGLDevice(cudaDeviceHandle));

	// Set up assorted OpenGL state
	const GLfloat background[] =
	{ params.background.r, params.background.g, params.background.b, 1.0f };
	if (params.mode != ColoringMode::BlendedColor && params.mode != ColoringMode::CentroidColor)
		glPolygonMode(GL_FRONT_AND_BACK, GL_LINE);
	glDisable(GL_DEPTH_TEST);
	glDisable(GL_STENCIL_TEST);
	glEnable(GL_TEXTURE_2D);
	glViewport(0, 0, dims.x, dims.y);

	// Compile and install the shaders for use with OpenGL
	GLuint vertShader, fragShader, glProgram;
	GLint compileStatus;
	vertShader = glCreateShader(GL_VERTEX_SHADER);
	glShaderSource(vertShader, 1, vertexShaderSource, NULL);
	glCompileShader(vertShader);
	glGetShaderiv(vertShader, GL_COMPILE_STATUS, &compileStatus);
	if (compileStatus != GL_TRUE)
		return PRINT_ERR_MSG(GL_COMPILE_ERROR);
	fragShader = glCreateShader(GL_FRAGMENT_SHADER);
	glShaderSource(fragShader, 1, fragmentShaderSource, NULL);
	glCompileShader(fragShader);
	glGetShaderiv(fragShader, GL_COMPILE_STATUS, &compileStatus);
	if (compileStatus != GL_TRUE)
		return PRINT_ERR_MSG(GL_COMPILE_ERROR);
	glProgram = glCreateProgram();
	glAttachShader(glProgram, vertShader);
	glAttachShader(glProgram, fragShader);
	glLinkProgram(glProgram);
	glValidateProgram(glProgram);
	GLint linkStatus, validateStatus;
	glGetProgramiv(glProgram, GL_LINK_STATUS, &linkStatus);
	glGetProgramiv(glProgram, GL_VALIDATE_STATUS, &validateStatus);
	if (linkStatus != GL_TRUE)
		return PRINT_ERR_MSG(GL_LINK_ERROR);
	if (validateStatus != GL_TRUE)
		return PRINT_ERR_MSG(GL_VALIDATE_ERROR);
	glDeleteShader(vertShader);
	glDeleteShader(fragShader);
	glRetCode = glGetError();
	if (glRetCode != GL_NO_ERROR)
		return PRINT_ERR_MSG_GL(glRetCode);

	// Generate a framebuffer for OpenGL to draw to
	GLuint dev_texture, dev_frameBuf;
	glGenTextures(1, &dev_texture);
	glBindTexture(GL_TEXTURE_RECTANGLE, dev_texture);
	glTexParameterf(GL_TEXTURE_RECTANGLE, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
	glTexParameterf(GL_TEXTURE_RECTANGLE, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
	glTexImage2D(GL_TEXTURE_RECTANGLE, 0, GL_RGB8, dims.x, dims.y, 0, GL_RGB, GL_UNSIGNED_BYTE, NULL);
	glGenFramebuffers(1, &dev_frameBuf);
	glBindFramebuffer(GL_FRAMEBUFFER, dev_frameBuf);
	glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_RECTANGLE, dev_texture, 0);
	if (glCheckFramebufferStatus(GL_FRAMEBUFFER) != GL_FRAMEBUFFER_COMPLETE)
		return PRINT_ERR_MSG_GL(glGetError());
	glRetCode = glGetError();
	if (glRetCode != GL_NO_ERROR)
		return PRINT_ERR_MSG_GL(glRetCode);

	// Generate a vertex array object for OpenGL to use (with data sourced from CUDA)
	GLuint dev_vao, dev_vertexBuffer;
	glCreateVertexArrays(1, &dev_vao);
	glBindVertexArray(dev_vao);
	glGenBuffers(1, &dev_vertexBuffer);
	glBindBuffer(GL_ARRAY_BUFFER, dev_vertexBuffer);
	glNamedBufferStorage(dev_vertexBuffer, dims.x * dims.y / 2 * sizeof(Triangle), NULL, 0);
	glRetCode = glGetError();
	if (glRetCode != GL_NO_ERROR)
		return PRINT_ERR_MSG_GL(glRetCode);
	glVertexArrayAttribBinding(dev_vao, 0, 0);
	glVertexArrayAttribFormat(dev_vao, 0, 2, GL_FLOAT, GL_FALSE, offsetof(Point, x));
	glEnableVertexArrayAttrib(dev_vao, 0);
	glVertexArrayVertexBuffer(dev_vao, 0, dev_vertexBuffer, 0, sizeof(Point));
	glVertexArrayAttribBinding(dev_vao, 1, 0);
	glVertexArrayAttribFormat(dev_vao, 1, 3, GL_FLOAT, GL_FALSE, offsetof(Point, color));
	glEnableVertexArrayAttrib(dev_vao, 1);
	glVertexArrayVertexBuffer(dev_vao, 0, dev_vertexBuffer, 0, sizeof(Point));
	glRetCode = glGetError();
	if (glRetCode != GL_NO_ERROR)
		return PRINT_ERR_MSG_GL(glRetCode);

	if (params.debug)
	{
		cudaGetDeviceProperties(&prop, cudaDeviceHandle);
		printDeviceData(prop);
	}

	// Allocate host memory
	auto host_imgBuf = reinterpret_cast<Image>(malloc(imgBufSize * sizeof(Pixel)));
	uint32_t host_numTriangles;

	// Allocate CUDA resources
	Image dev_imgBuf;
	ChannelBuf dev_intensities, dev_edges, dev_pointHistoricity;
	float* dev_distr;
	float* dev_distrRows;
	float* dev_distrRowSums;
	float* dev_distrCols;
	Point* dev_pointBuf;
	uint32_t* dev_voronoi;
	uint32_t* dev_numTriangles;
	curandState_t* dev_curandStates;
	CUDA_CALL(cudaMalloc(&dev_imgBuf, imgBufSize * sizeof(Pixel)));
	CUDA_CALL(cudaMalloc(&dev_intensities, imgBufSize * sizeof(channel_t)));
	CUDA_CALL(cudaMalloc(&dev_edges, imgBufSize * sizeof(channel_t)));
	CUDA_CALL(cudaMalloc(&dev_pointHistoricity, imgBufSize * sizeof(channel_t)));
	CUDA_CALL(cudaMalloc(&dev_distr, imgBufSize * sizeof(float)));
	CUDA_CALL(cudaMalloc(&dev_distrRows, imgBufSize * sizeof(float)));
	CUDA_CALL(cudaMalloc(&dev_distrRowSums, dims.y * sizeof(float)));
	CUDA_CALL(cudaMalloc(&dev_distrCols, dims.y * sizeof(float)));
	CUDA_CALL(cudaMalloc(&dev_pointBuf, numPoints * sizeof(Point)));
	CUDA_CALL(cudaMalloc(&dev_voronoi, imgBufSize * sizeof(uint32_t)));
	CUDA_CALL(cudaMalloc(&dev_numTriangles, sizeof(uint32_t)));
	CUDA_CALL(cudaMalloc(&dev_curandStates, numPoints * sizeof(curandState_t)));
	retCode = preallocBlockSums(imgBufSize);
	if (retCode != SUCCESS)
		return retCode;

	// Hook up the OpenGL vertex buffer to CUDA
	cudaGraphicsResource* dev_vertBufCUDA;
	CUDA_CALL(cudaGraphicsGLRegisterBuffer(&dev_vertBufCUDA, dev_vertexBuffer, cudaGLMapFlagsNone));

	// Initialize state for cuRAND and historicity
	CUDA_CALL(cudaMemset(dev_pointHistoricity, 0, imgBufSize * sizeof(channel_t)));
	cuRAND_init<<<blockSizePoints, threadSize>>>(dev_curandStates, params.seed);

	// Create events to be used if debugging is enabled
	cudaEvent_t start, distrGenerated, pointsSampled, pointsTesselated, end;
	if (params.timing)
	{
		CUDA_CALL(cudaEventCreate(&start));
		CUDA_CALL(cudaEventCreate(&distrGenerated));
		CUDA_CALL(cudaEventCreate(&pointsSampled));
		CUDA_CALL(cudaEventCreate(&pointsTesselated));
		CUDA_CALL(cudaEventCreate(&end));
	}

	// Perform processing on the image/video
	for (auto i = 0; i < input.getFrameCount(); ++i)
	{
		// Copy input frame to the GPU
		auto imgPtr = input.getFrame(i);
		if (imgPtr == nullptr)
			break;
		CUDA_CALL(cudaMemcpy(dev_imgBuf, imgPtr, imgBufSize * sizeof(Pixel),
			cudaMemcpyHostToDevice));

		if (params.timing)
			CUDA_CALL(cudaEventRecord(start, 0));

		// Generate distribution
		computePixelIntensities<<<blockSize, threadSize>>>(dev_imgBuf, dev_intensities, dims);
		detectEdges<<<blockSize, threadSize>>>(dev_intensities, dev_edges, dims);
		blendDistributionData<<<blockSize, threadSize>>>(dev_intensities, dev_edges,
			dev_pointHistoricity, dev_distr, dims, params.intensityEdgeWeight,
			params.historicityWeight);
		retCode = generateCDF(dev_distr, dev_distrRows, dev_distrCols, dev_distrRowSums, dims,
			blockSize, threadSize);
		if (retCode != SUCCESS)
			return retCode;

		if (params.timing)
			CUDA_CALL(cudaEventRecord(distrGenerated, 0));
		if (params.debug)
			visualizeDistr(dev_distr, dims);

		// Sample numPoints points from the distribution and assign point colors
		CUDA_CALL(cudaMemset(dev_pointHistoricity, 0, imgBufSize * sizeof(channel_t)));
		samplePoints<<<blockSizePoints, threadSize>>>(dev_distrRows, dev_distrCols, dims,
			dev_curandStates, dev_pointBuf, numPoints, dev_pointHistoricity);

		if (params.timing)
			CUDA_CALL(cudaEventRecord(pointsSampled, 0));

		// Perform triangulation of points
		CUDA_CALL(cudaGraphicsMapResources(1, &dev_vertBufCUDA));
		Triangle* dev_triangleBuf;
		size_t numTriangleBytes;
		CUDA_CALL(cudaGraphicsResourceGetMappedPointer((void**)&dev_triangleBuf, &numTriangleBytes,
			dev_vertBufCUDA));
		computeVoronoi<<<blockSize, threadSize>>>(dev_voronoi, dims, dev_pointBuf, numPoints);
		convertToGLCoords<<<blockSizePoints, threadSize>>>(dev_pointBuf, dims);
		CUDA_CALL(cudaMemset(dev_numTriangles, 0, sizeof(uint32_t)));
		constructTriangulation<<<blockSize, threadSize>>>(dev_voronoi, dims, dev_pointBuf,
			dev_triangleBuf, dev_numTriangles);
		assignTriangleColorings<<<blockSize, threadSize>>>(dev_triangleBuf, dev_numTriangles,
			dev_imgBuf, params.mode, params.background, params.foreground, dims);
		// TODO Investigate illegal memory access which is happening here
		CUDA_CALL(cudaMemcpy(&host_numTriangles, dev_numTriangles, sizeof(uint32_t),
			cudaMemcpyDeviceToHost));
		CUDA_CALL(cudaGraphicsUnmapResources(1, &dev_vertBufCUDA));

		if (params.debug)
			std::cout << "Number of triangles in frame " << i << ": " << host_numTriangles
				<< std::endl << std::endl;
		if (params.timing)
			CUDA_CALL(cudaEventRecord(pointsTesselated, 0));

		// Call out to OpenGL to generate the final image
		glBindFramebuffer(GL_FRAMEBUFFER, dev_frameBuf);
		glClearBufferfv(GL_COLOR, 0, background);
		glUseProgram(glProgram);
		glDrawArrays(GL_TRIANGLES, 0, host_numTriangles * 3);
		glRetCode = glGetError();
		if (glRetCode != GL_NO_ERROR)
			return PRINT_ERR_MSG_GL(glRetCode);

		if (params.timing)
		{
			CUDA_CALL(cudaEventRecord(end, 0));
			CUDA_CALL(cudaEventSynchronize(end));
		}

		// Copy the resulting frame back to the CPU
		glReadBuffer(GL_COLOR_ATTACHMENT0);
		glReadPixels(0, 0, dims.x, dims.y, GL_BGR, GL_UNSIGNED_BYTE, host_imgBuf);
		output.addFrame(host_imgBuf, dims.x, dims.y);

		if (params.timing)
		{
			retCode = printTimings(start, distrGenerated, pointsSampled, pointsTesselated, end, i);
			if (retCode != SUCCESS)
				return retCode;
		}
	}

	// Release resources
	CUDA_CALL(cudaGraphicsUnregisterResource(dev_vertBufCUDA));
	glDeleteProgram(glProgram);
	glDeleteVertexArrays(1, &dev_vao);
	glDeleteBuffers(1, &dev_vertexBuffer);
	glDeleteFramebuffers(1, &dev_frameBuf);
	glDeleteTextures(1, &dev_texture);
	glutDestroyWindow(glutWindow);
	free(host_imgBuf);
	CUDA_CALL(cudaFree(dev_imgBuf));
	CUDA_CALL(cudaFree(dev_intensities));
	CUDA_CALL(cudaFree(dev_edges));
	CUDA_CALL(cudaFree(dev_pointHistoricity));
	CUDA_CALL(cudaFree(dev_distr));
	CUDA_CALL(cudaFree(dev_distrRows));
	CUDA_CALL(cudaFree(dev_distrRowSums));
	CUDA_CALL(cudaFree(dev_distrCols));
	CUDA_CALL(cudaFree(dev_pointBuf));
	CUDA_CALL(cudaFree(dev_voronoi));
	CUDA_CALL(cudaFree(dev_numTriangles));
	CUDA_CALL(cudaFree(dev_curandStates));
	deallocBlockSums();

	return SUCCESS;
}


__global__ void cuRAND_init(curandState_t* states, uint64_t seed)
{
	short x = threadIdx.x + blockIdx.x * blockDim.x;
	short y = threadIdx.y + blockIdx.y * blockDim.y;
	uint32_t offset = x + (y * (gridDim.x * blockDim.x));

	// Perform initialization for random number generation
	//   Copying the curandState to local storage dramatically improves performance
	curandState state = states[offset];
	curand_init(seed + x, y, 0, &state);
	states[offset] = state;
}


__global__ void computePixelIntensities(Image img, ChannelBuf intensities, short2 dims)
{
	short x = threadIdx.x + blockIdx.x * blockDim.x;
	short y = threadIdx.y + blockIdx.y * blockDim.y;
	uint32_t offset = x + (y * dims.x);
	if (x > dims.x || y > dims.y)
		return;

	intensities[offset] = (img[offset].r + img[offset].g + img[offset].b) / 3;
}


__global__ void detectEdges(ChannelBuf intensities, ChannelBuf edges, short2 dims)
{
	// 3x3 edge detection filter
	const float edgeFilter[3][3] =
	{
		{ -0.333f, -0.333f, 0.000f },
		{ -0.333f, 0.000f, 0.333f },
		{ 0.000f, 0.333f, 0.333f },
	};

	short x = threadIdx.x + (blockIdx.x * blockDim.x);
	short y = threadIdx.y + (blockIdx.y * blockDim.y);
	unsigned int offset = x + (y * dims.x);
	if (x > dims.x || y > dims.y)
		return;

	// Compute image gradient
	float val = 0;
	for (int i = offset - dims.x, ii = 0; i <= offset + dims.x; i += dims.x, ++ii)
	{
		for (int j = i - 1, jj = 0; j <= i + 1; ++j, ++jj)
		{
			if (j >= 0 && j < dims.x * dims.y)
			{
				val += edgeFilter[ii][jj] * intensities[j];
			}
		}
	}

	// Save the calculated edge intensity with light filtering
	edges[offset] = (channel_t)abs(val) > EDGE_THRESH ? abs(val) : 0;
}


__global__ void blendDistributionData(ChannelBuf intensity, ChannelBuf edges,
	ChannelBuf historicity, float* distribution, short2 dims, float intEdgeWeight,
	float historicityWeight)
{
	short x = threadIdx.x + blockIdx.x * blockDim.x;
	short y = threadIdx.y + blockIdx.y * blockDim.y;
	uint32_t offset = x + (y * dims.x);
	if (x > dims.x || y > dims.y)
		return;

	// Calculate the value for this point in the distribution
	distribution[offset] = (intEdgeWeight*intensity[offset]
		+ (1 - intEdgeWeight)*edges[offset]) / 255;
	distribution[offset] = historicityWeight*historicity[offset]
		+ (1 - historicityWeight)*distribution[offset];
}


ReturnCode generateCDF(float* distribution, float* distrRows, float* distrCols,
	float* distrRowSums, short2 dims, dim3 blockSize, dim3 threadSize)
{
	cudaError_t cudaRetCode;

	for (auto i = 0; i < dims.y - 1; ++i)
		prescanArray(distrRows + dims.x*i, distribution + dims.x*i, dims.x);
	copyRowSums<<<blockSize, threadSize>>>(distrRows, distrRowSums, dims);
	prescanArray(distrCols, distrRowSums, dims.y);
	normalizeCDF<<<blockSize, threadSize>>>(distrRows, distrRowSums, distrCols, dims);
	float one = 1;
	CUDA_CALL(cudaMemcpy(distrCols + dims.y - 1, &one, sizeof(one), cudaMemcpyHostToDevice));

	return SUCCESS;
}


__global__ void copyRowSums(float* distrRows, float* distrRowSums, short2 dims)
{
	short x = threadIdx.x + blockIdx.x * blockDim.x;
	short y = threadIdx.y + blockIdx.y * blockDim.y;
	uint32_t offset = x + (y * dims.x);
	if (offset > dims.x*dims.y || x != dims.x - 1)
		return;

	distrRowSums[y] = distrRows[offset];
}


__global__ void normalizeCDF(float* distrRows, float* distrRowSums, float* distrCols, short2 dims)
{
	short x = threadIdx.x + blockIdx.x * blockDim.x;
	short y = threadIdx.y + blockIdx.y * blockDim.y;
	uint32_t offset = x + (y * dims.x);
	if (offset > dims.x*dims.y)
		return;

	distrRows[offset] /= distrRowSums[y];
	if (offset < dims.y - 1)
		distrCols[offset] /= distrCols[dims.y - 1];
}


__device__ uint16_t binarySearch(float* arr, size_t arrSize, uint32_t minIndex, uint32_t maxIndex,
	float target)
{
	uint32_t idx = minIndex + (maxIndex - minIndex) / 2;
	if (arr[idx] <= target && (idx >= arrSize - 1 || arr[idx + 1] >= target))
		return idx;
	if (arr[idx] > target)
		return binarySearch(arr, arrSize, minIndex, idx - 1, target);
	return binarySearch(arr, arrSize, idx + 1, maxIndex, target);
}


__global__ void samplePoints(float* rowDist, float* colDist, short2 dims,
	curandState_t* randStates, Point* pointBuf, uint32_t numPoints, ChannelBuf historicityBuf)
{
	short x = threadIdx.x + blockIdx.x * blockDim.x;
	short y = threadIdx.y + blockIdx.y * blockDim.y;
	uint32_t offset = x + (y * (gridDim.x * blockDim.x));

	// Generate two random numbers to use for index caluculation
	//   Copying the curandState to local storage dramatically improves performance
	curandState state = randStates[offset];
	auto randX = 1 - curand_uniform(&state);
	auto randY = 1 - curand_uniform(&state);
	randStates[offset] = state;

	// Perform binary searches to find the corresponding index in the distribution
	pointBuf[offset].y = binarySearch(colDist, dims.y, 0, dims.y, randY);
	pointBuf[offset].x = binarySearch(rowDist + ((uint16_t)pointBuf[offset].y * dims.x), dims.x, 0,
		dims.x, randX);

	// 'Paint' the selected point into historicityBuf with atomicAdd
	// TODO
}


__global__ void computeVoronoi(uint32_t* voronoi, short2 dims, Point* points, uint32_t numPoints)
{
	short x = threadIdx.x + blockIdx.x * blockDim.x;
	short y = threadIdx.y + blockIdx.y * blockDim.y;
	uint32_t offset = x + (y * (gridDim.x * blockDim.x));
	if (x > dims.x || y > dims.y)
		return;

	// Find the closest point
	short deltaX, deltaY;
	float dist;
	float minDist = FLT_MAX;
	uint32_t bestIdx = 0;
	for (uint32_t i = 0; i < numPoints; ++i)
	{
		deltaX = points[i].x - x;
		deltaY = points[i].y - y;
		dist = deltaX*deltaX + deltaY*deltaY;
		if (dist < minDist)
		{
			minDist = dist;
			bestIdx = i;
		}
	}

	// Store the index of the closest point into Voronoi diagram
	voronoi[offset] = bestIdx;
}


__global__ void convertToGLCoords(Point* points, short2 dims)
{
	short x = threadIdx.x + blockIdx.x * blockDim.x;
	short y = threadIdx.y + blockIdx.y * blockDim.y;
	uint32_t offset = x + (y * (gridDim.x * blockDim.x));

	points[offset].x /= (dims.x / 2);
	points[offset].y /= (dims.y / 2);

	points[offset].x -= 1;
	points[offset].y -= 1;
}


__global__ void constructTriangulation(uint32_t* voronoi, short2 dims, Point* points,
	Triangle* triangulation, uint32_t* numTriangles)
{
	short x = threadIdx.x + blockIdx.x * blockDim.x;
	short y = threadIdx.y + blockIdx.y * blockDim.y;
	uint32_t offset = x + (y * (gridDim.x * blockDim.x));
	if (x > dims.x - 2 || y > dims.y - 2)
		return;

	// Determine if the current point is a Voronoi vertex
	//   First, check for diagonal connections (if a point is 8-connected, it's not a vertex)
	if (voronoi[offset] == voronoi[offset + dims.x + 1] ||
		voronoi[offset + 1] == voronoi[offset + dims.x])
		return;
	//   Second, count the number of neighbors--if it's 3 or more, this is a vetex
	uint8_t numNeighbors = 1;
	uint32_t neighbors[4];
	neighbors[0] = voronoi[offset];
	if (voronoi[offset] != voronoi[offset + 1])
	{
		neighbors[numNeighbors] = voronoi[offset + 1];
		++numNeighbors;
	}
	if (voronoi[offset] != voronoi[offset + dims.x])
	{
		neighbors[numNeighbors] = voronoi[offset + dims.x];
		++numNeighbors;
	}
	if (voronoi[offset + dims.x] != voronoi[offset + dims.x + 1])
	{
		neighbors[numNeighbors] = voronoi[offset + dims.x + 1];
		++numNeighbors;
	}

	// If a point is a vertex, add triangles connecting the neighboring points
	if (numNeighbors == 3)
	{
		auto index = atomicAdd(numTriangles, 1);

		triangulation[index].p1 = points[neighbors[0]];
		triangulation[index].p2 = points[neighbors[2]];
		triangulation[index].p3 = points[neighbors[1]];
	}
	else if (numNeighbors == 4)
	{
		auto index = atomicAdd(numTriangles, 2);

		triangulation[index].p1 = points[neighbors[0]];
		triangulation[index].p2 = points[neighbors[2]];
		triangulation[index].p3 = points[neighbors[1]];

		triangulation[index + 1].p1 = points[neighbors[2]];
		triangulation[index + 1].p2 = points[neighbors[3]];
		triangulation[index + 1].p3 = points[neighbors[1]];
	}
}


__global__ void assignTriangleColorings(Triangle* triangles, uint32_t* numTriangles, Image srcImg,
	ColoringMode mode, Color bgColor, Color fgColor, short2 dims)
{
	short x = threadIdx.x + blockIdx.x * blockDim.x;
	short y = threadIdx.y + blockIdx.y * blockDim.y;
	uint32_t offset = x + (y * (gridDim.x * blockDim.x));
	if (offset >= *numTriangles)
		return;

	switch (mode)
	{
	case ColoringMode::CentroidColor:
		uint16_t centroidX = dims.x / 6 *
			(triangles[offset].p1.x + triangles[offset].p2.x + triangles[offset].p3.x + 3);
		uint16_t centroidY = dims.y / 6 *
			(triangles[offset].p1.y + triangles[offset].p2.y + triangles[offset].p3.y + 3);
		uint32_t centroidOffset = centroidX + (centroidY * dims.x);

		triangles[offset].p1.color.r = srcImg[centroidOffset].b / 255.0;
		triangles[offset].p1.color.g = srcImg[centroidOffset].g / 255.0;
		triangles[offset].p1.color.b = srcImg[centroidOffset].r / 255.0;

		triangles[offset].p2.color.r = srcImg[centroidOffset].b / 255.0;
		triangles[offset].p2.color.g = srcImg[centroidOffset].g / 255.0;
		triangles[offset].p2.color.b = srcImg[centroidOffset].r / 255.0;

		triangles[offset].p3.color.r = srcImg[centroidOffset].b / 255.0;
		triangles[offset].p3.color.g = srcImg[centroidOffset].g / 255.0;
		triangles[offset].p3.color.b = srcImg[centroidOffset].r / 255.0;
		break;
	case ColoringMode::BlendedColor:
	case ColoringMode::PixelColors:
		uint32_t imgOffset;
		uint16_t pixX, pixY;
		
		pixX = dims.x / 2 * (triangles[offset].p1.x + 1);
		pixY = dims.y / 2 * (triangles[offset].p1.y + 1);
		imgOffset = pixX + (pixY * dims.x);
		triangles[offset].p1.color.r = srcImg[imgOffset].b / 255.0;
		triangles[offset].p1.color.g = srcImg[imgOffset].g / 255.0;
		triangles[offset].p1.color.b = srcImg[imgOffset].r / 255.0;

		pixX = dims.x / 2 * (triangles[offset].p2.x + 1);
		pixY = dims.y / 2 * (triangles[offset].p2.y + 1);
		imgOffset = pixX + (pixY * dims.x);
		triangles[offset].p2.color.r = srcImg[imgOffset].b / 255.0;
		triangles[offset].p2.color.g = srcImg[imgOffset].g / 255.0;
		triangles[offset].p2.color.b = srcImg[imgOffset].r / 255.0;

		pixX = dims.x / 2 * (triangles[offset].p3.x + 1);
		pixY = dims.y / 2 * (triangles[offset].p3.y + 1);
		imgOffset = pixX + (pixY * dims.x);
		triangles[offset].p3.color.r = srcImg[imgOffset].b / 255.0;
		triangles[offset].p3.color.g = srcImg[imgOffset].g / 255.0;
		triangles[offset].p3.color.b = srcImg[imgOffset].r / 255.0;
		break;
	case ColoringMode::SolidColors:
		triangles[offset].p1.color = fgColor;
		triangles[offset].p2.color = fgColor;
		triangles[offset].p3.color = fgColor;
		break;
	}
}