/*
 * TODO
 */


#include <math.h>
#include <windows.h>

// OpenGL
#include <GL\glew.h>
#include <GL\GL.h>
#include <GL\glut.h>

// CUDA
#include <cuda_runtime.h>
#include <cuda_gl_interop.h>
#include <curand_kernel.h>
#include "scan.cu"

#include "constants.hpp"
#include "imageProcessor.hpp"
#include "util.hpp"


// Forward declarations
__global__ void cuRAND_init(curandState_t* states, uint64_t seed);

__global__ void computePixelIntensities(Image img, ChannelBuf intensities, short2 dims);
__global__ void detectEdges(ChannelBuf intensities, ChannelBuf edges, short2 dims);
__global__ void blendDistributionData(ChannelBuf intensity, ChannelBuf edges,
    ChannelBuf historicity, float* distribution, short2 dims, float intEdgeWeight,
    float historicityWeight);
ReturnCode generateCDF(float* distribution, float* distrRows, float* distrCols,
    float* dev_distrRowSums, short2 dims, dim3 blockSize, dim3 threadSize);
__global__ void copyRowSums(float* distrRows, float* distrRowSums, short2 dims);
__global__ void normalizeCDF(float* distrRows, float* distrRowSums, float* distrCols, short2 dims);

__device__ uint16_t binarySearch(float* arr, short2 dims, uint32_t minIndex, uint32_t maxIndex,
	float target);
__global__ void samplePoints(float* rowDist, float* colDist, short2 dims,
	curandState_t* randStates, Point* pointBuf, uint32_t numPoints, ChannelBuf historicityBuf);


// Shader code
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
	uint32_t numTriangles = 2 * numPoints;
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
	const GLfloat background[] = { params.background.r / 255.0f, params.background.g / 255.0f,
		params.background.b / 255.0f, 1.0f };
	glPolygonMode(GL_FRONT_AND_BACK, GL_LINE);
	glDisable(GL_DEPTH_TEST);
	glEnable(GL_TEXTURE_2D);
	glRetCode = glGetError();
	if (glRetCode != GL_NO_ERROR)
		return PRINT_ERR_MSG_GL(glRetCode);

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
	glBindTexture(GL_TEXTURE_2D, dev_texture);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
	glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB8, dims.x, dims.y, 0, GL_RGB, GL_UNSIGNED_INT, NULL);
	glGenFramebuffers(1, &dev_frameBuf);
	glBindFramebuffer(GL_FRAMEBUFFER, dev_frameBuf);
	glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, dev_texture, 0);
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
	glNamedBufferStorage(dev_vertexBuffer, numTriangles * sizeof(Triangle), NULL, 0);
	glRetCode = glGetError();
	if (glRetCode != GL_NO_ERROR)
		return PRINT_ERR_MSG_GL(glRetCode);
	glVertexArrayAttribBinding(dev_vao, 0, 0);
	glVertexArrayAttribFormat(dev_vao, 0, 2, GL_UNSIGNED_SHORT, GL_FALSE, offsetof(Point, x));
	glEnableVertexArrayAttrib(dev_vao, 0);
	glVertexArrayAttribBinding(dev_vao, 1, 0);
	glVertexArrayAttribFormat(dev_vao, 1, 3, GL_UNSIGNED_BYTE, GL_FALSE, offsetof(Point, color));
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

    // Allocate memory
    Image dev_imgBuf;
    ChannelBuf dev_intensities, dev_edges, dev_pointHistoricity;
    float* dev_distr;
	float* dev_distrRows;
    float* dev_distrRowSums;
    float* dev_distrCols;
	Point* dev_pointBuf;
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
	CUDA_CALL(cudaMalloc(&dev_curandStates, numPoints * sizeof(curandState_t)));
    auto host_imgBuf = reinterpret_cast<Image>(malloc(imgBufSize * sizeof(Pixel)));
	retCode = preallocBlockSums(imgBufSize);
	if (retCode != SUCCESS)
		return retCode;

    CUDA_CALL(cudaMemset(dev_pointHistoricity, 0, imgBufSize * sizeof(channel_t)));

    // Initialize state for cuRAND
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
		
        // Sample numPoints points from the distribution
		CUDA_CALL(cudaMemset(dev_pointHistoricity, 0, imgBufSize * sizeof(channel_t)));
		samplePoints<<<blockSizePoints, threadSize>>>(dev_distrRows, dev_distrCols, dims,
            dev_curandStates, dev_pointBuf, numPoints, dev_pointHistoricity);
        
        if (params.timing)
            CUDA_CALL(cudaEventRecord(pointsSampled, 0));

        // Perform tesselation of points
        //   Follow the paper linked to in the project proposal
		//   Also assign point colors based on the current coloring mode
        
        if (params.timing)
            CUDA_CALL(cudaEventRecord(pointsTesselated, 0));

        // Call out to OpenGL to generate the final image
		glBindFramebuffer(GL_FRAMEBUFFER, dev_frameBuf);
		glClearNamedFramebufferfv(dev_frameBuf, GL_COLOR, 0, background);
		glUseProgram(glProgram);
		glDrawArrays(GL_POINTS, 0, numTriangles * 3);
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
        { -0.333f, -0.333f,  0.000f },
        { -0.333f,  0.000f,  0.333f },
        {  0.000f,  0.333f,  0.333f },
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
    edges[offset] = (channel_t) abs(val) > EDGE_THRESH ? abs(val) : 0;
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
	pointBuf[offset].x = binarySearch(rowDist + (pointBuf[offset].y * dims.x), dims.x, 0, dims.x,
		randX);

    // 'Paint' the selected point into historicityBuf with atomicAdd
    // TODO
}