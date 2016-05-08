/*
 * TODO
 */


#include <ctime>
#include <math.h>

// CUDA
#include <cuda_runtime.h>
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
void generateCDF(float* distribution, float* distrRows, float* distrCols,
    float* dev_distrRowSums, short2 dims, dim3 blockSize, dim3 threadSize);
__global__ void copyRowSums(float* distrRows, float* distrRowSums, short2 dims);
__global__ void normalizeCDF(float* distrRows, float* distrRowSums, float* distrCols, short2 dims);

__device__ uint16_t binarySearch(float* arr, short2 dims, uint32_t minIndex, uint32_t maxIndex,
	float target);
__global__ void samplePoints(float* rowDist, float* colDist, short2 dims,
    curandState_t* randStates, short2* pointBuf, uint32_t numPoints, ChannelBuf historicityBuf);


__global__ void genDebugImg(Image img, short2* points, short2 dims, uint32_t numPoints);


ReturnCode processImageResource(ImageResource& input, ImageResource& output, ParamBundle params)
{
    cudaError_t cudaRetCode;

    // Prepare relevant state
    auto dims = make_short2(input.getWidth(), input.getHeight());
    uint32_t imgBufSize = dims.x * dims.y;
    uint32_t numPoints = static_cast<uint32_t>(imgBufSize * params.pointRatio);
	uint32_t numTriangles = numPoints + 1; // TODO Figure out what this actually needs to be
    dim3 blockSize(dims.x / THREADS_PER_BLOCK, dims.y / THREADS_PER_BLOCK);
	auto pointDim = static_cast<uint32_t>(sqrt(numPoints / THREADS_PER_BLOCK / THREADS_PER_BLOCK));
	dim3 blockSizePoints(pointDim, pointDim);
    dim3 threadSize(THREADS_PER_BLOCK, THREADS_PER_BLOCK);

    // Allocate memory
    Image dev_imgBuf;
    ChannelBuf dev_intensities, dev_edges, dev_pointHistoricity;
    float* dev_distr;
	float* dev_distrRows;
    float* dev_distrRowSums;
    float* dev_distrCols;
    short2* dev_pointBuf;
	short2* dev_triangleBuf;
    curandState_t* dev_curandStates;
    CUDA_CALL(cudaMalloc(&dev_imgBuf, imgBufSize * sizeof(Pixel)));
    CUDA_CALL(cudaMalloc(&dev_intensities, imgBufSize * sizeof(channel_t)));
    CUDA_CALL(cudaMalloc(&dev_edges, imgBufSize * sizeof(channel_t)));
    CUDA_CALL(cudaMalloc(&dev_pointHistoricity, imgBufSize * sizeof(channel_t)));
	CUDA_CALL(cudaMalloc(&dev_distr, imgBufSize * sizeof(float)));
	CUDA_CALL(cudaMalloc(&dev_distrRows, imgBufSize * sizeof(float)));
    CUDA_CALL(cudaMalloc(&dev_distrRowSums, dims.y * sizeof(float)));
    CUDA_CALL(cudaMalloc(&dev_distrCols, dims.y * sizeof(float)));
	CUDA_CALL(cudaMalloc(&dev_pointBuf, numPoints * sizeof(short2)));
	CUDA_CALL(cudaMalloc(&dev_triangleBuf, numTriangles * sizeof(short2) * 3));
	CUDA_CALL(cudaMalloc(&dev_curandStates, numPoints * sizeof(curandState_t)));
    auto host_imgBuf = reinterpret_cast<Image>(malloc(imgBufSize * sizeof(Pixel)));
	preallocBlockSums(imgBufSize);

    CUDA_CALL(cudaMemset(dev_pointHistoricity, 0, imgBufSize * sizeof(channel_t)));

    // Initialize state for cuRAND
	// cuRAND_init<<<blockSizePoints, threadSize>>>(dev_curandStates, params.debug ? 0 : time(NULL));
	cuRAND_init<<<blockSizePoints, threadSize>>>(dev_curandStates, time(NULL));

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
		generateCDF(dev_distr, dev_distrRows, dev_distrCols, dev_distrRowSums, dims,
			blockSize, threadSize);
        
        if (params.timing)
            CUDA_CALL(cudaEventRecord(distrGenerated, 0));
		if (params.debug)
			visualizeDistr(dev_distr, dims);
		
        // Sample numPoints points from the distribution
		CUDA_CALL(cudaMemset(dev_pointHistoricity, 0, imgBufSize * sizeof(channel_t)));
		samplePoints<<<blockSizePoints, threadSize>>>(dev_distrRows, dev_distrCols, dims,
            dev_curandStates, dev_pointBuf, numPoints, dev_pointHistoricity);
        
		if (params.debug)
		{
			genDebugImg<<<blockSize, threadSize>>>(dev_imgBuf, dev_pointBuf, dims, numPoints);
			displayPreviewImageGPU(dev_imgBuf, dims);
		}
        if (params.timing)
            CUDA_CALL(cudaEventRecord(pointsSampled, 0));

        // Perform tesselation of points
        //   Follow the paper linked to in the project proposal
        
        if (params.timing)
            CUDA_CALL(cudaEventRecord(pointsTesselated, 0));

        // Call out to OpenGL to generate the final image; store it in dev_imgBuf
        //   TODO
        
        if (params.timing)
        {
            CUDA_CALL(cudaEventRecord(end, 0));
            CUDA_CALL(cudaEventSynchronize(end));
        }

        // Copy the resulting frame back to the CPU
        CUDA_CALL(cudaMemcpy(host_imgBuf, dev_imgBuf, imgBufSize * sizeof(Pixel), cudaMemcpyDeviceToHost));
        output.addFrame(host_imgBuf, dims.x, dims.y);
        if (params.timing)
            printTimings(start, distrGenerated, pointsSampled, pointsTesselated, end, i);
    }

    // Free memory
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
	CUDA_CALL(cudaFree(dev_triangleBuf));
    CUDA_CALL(cudaFree(dev_curandStates));
	deallocBlockSums();

    return NOT_YET_IMPLEMENTED;
}


__global__ void cuRAND_init(curandState_t* states, uint64_t seed)
{
    // The x and y coordinates for which this instance of the kernel is responsible
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
    // The x and y coordinates for which this instance of the kernel is responsible
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

    // The x and y coordinates for which this instance of the kernel is responsible
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
    // The x and y coordinates for which this instance of the kernel is responsible
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


void generateCDF(float* distribution, float* distrRows, float* distrCols,
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
}


__global__ void copyRowSums(float* distrRows, float* distrRowSums, short2 dims)
{
    // The x and y coordinates for which this instance of the kernel is responsible
    short x = threadIdx.x + blockIdx.x * blockDim.x;
    short y = threadIdx.y + blockIdx.y * blockDim.y;
    uint32_t offset = x + (y * dims.x);
    if (offset > dims.x*dims.y || x != dims.x - 1)
        return;

    // Copy the maximum values of each row
    distrRowSums[y] = distrRows[offset];
}


__global__ void normalizeCDF(float* distrRows, float* distrRowSums, float* distrCols, short2 dims)
{
    // The x and y coordinates for which this instance of the kernel is responsible
    short x = threadIdx.x + blockIdx.x * blockDim.x;
    short y = threadIdx.y + blockIdx.y * blockDim.y;
    uint32_t offset = x + (y * dims.x);
    if (offset > dims.x*dims.y)
        return;

    // Normalize distrRows
    distrRows[offset] /= distrRowSums[y];

    // Normalize distrCols
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
    curandState_t* randStates, short2* pointBuf, uint32_t numPoints, ChannelBuf historicityBuf)
{
    // The x and y coordinates for which this instance of the kernel is responsible
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
	auto yVal = binarySearch(colDist, dims.y, 0, dims.y, randY);
	auto xVal = binarySearch(rowDist + (yVal * dims.x), dims.x, 0, dims.x, randX);

    // Convert the computed index into a point value
	pointBuf[offset] = make_short2(xVal, yVal);

    // 'Paint' the selected point into historicityBuf with atomicAdd
    // TODO
}


__global__ void genDebugImg(Image img, short2* points, short2 dims, uint32_t numPoints)
{
	// The x and y coordinates for which this instance of the kernel is responsible
	short x = threadIdx.x + blockIdx.x * blockDim.x;
	short y = threadIdx.y + blockIdx.y * blockDim.y;
	uint32_t offset = x + (y * dims.x);
	if (x > dims.x || y > dims.y)
		return;

	if (offset < numPoints)
	{
		uint32_t pointOffset = points[offset].x + (points[offset].y * dims.x);
		img[pointOffset].r = 0;
		img[pointOffset].g = 0;
		img[pointOffset].b = 255;
	}
}