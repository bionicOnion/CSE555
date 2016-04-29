/*
 * TODO
 */


// CUDA
#include <cuda_runtime.h>

#include "constants.hpp"
#include "imageProcessor.hpp"
#include "util.hpp"


// Forward declarations
__global__ void computePixelIntensities(Image img, ChannelBuf intensities, short2 dims);
__global__ void generateImagePyramid(ChannelBuf intensities, ChannelBuf pyramid, short2 dims);
__global__ void detectEdges(ChannelBuf imgPyramid, ChannelBuf edgesPyramid, short2 dims);
__global__ void blendDistributionData(ChannelBuf intensity, ChannelBuf edges,
	ChannelBuf historicity, double* distribution, short2 dims, float intEdgeWeight,
	float historicityWeight);
__global__ void generateCDF(double* distribution, size_t distrLen, short2 dims);
__global__ void samplePoints(double* distribution, size_t distrLen, short2* pointBuf,
	uint32_t numPoints, ChannelBuf historicityBuf, short2 dims);


// TODO add debug #ifdefs to look at distributiondouble*, Voronoi, etc.
ReturnCode processImageResource(ImageResource& input, ImageResource& output, ParamBundle params)
{
	cudaError_t cudaRetCode;

    // Prepare relevant state
    auto dims = make_short2(input.getWidth(), input.getHeight());
    uint32_t imgBufSize = dims.x * dims.y;
    uint32_t numPoints = static_cast<uint32_t>(imgBufSize * params.pointRatio);
    dim3 blockSize(dims.x / THREADS_PER_BLOCK, dims.y / THREADS_PER_BLOCK);
    dim3 threadSize(THREADS_PER_BLOCK, THREADS_PER_BLOCK);

    // Allocate memory
	Image dev_imgBuf;
	ChannelBuf dev_intensities, dev_imgPyramid, dev_edges, dev_pointHistoricity;
    double* dev_distr;
	short2* dev_pointBuf;
    CUDA_CALL(cudaMalloc(&dev_imgBuf, imgBufSize * sizeof(Pixel)));
	CUDA_CALL(cudaMalloc(&dev_intensities, imgBufSize * sizeof(channel_t)));
	CUDA_CALL(cudaMalloc(&dev_imgPyramid, imgBufSize * sizeof(channel_t) * 4 / 3));
	CUDA_CALL(cudaMalloc(&dev_edges, imgBufSize * sizeof(channel_t) * 4 / 3));
	CUDA_CALL(cudaMalloc(&dev_pointHistoricity, imgBufSize * sizeof(channel_t)));
    CUDA_CALL(cudaMalloc(&dev_distr, imgBufSize * sizeof(double)));
	CUDA_CALL(cudaMalloc(&dev_pointBuf, numPoints * sizeof(short2)));
	auto host_imgBuf = reinterpret_cast<Image>(malloc(imgBufSize * sizeof(Pixel)));

	CUDA_CALL(cudaMemset(dev_pointHistoricity, 0, imgBufSize * sizeof(channel_t)));

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
		CUDA_CALL(cudaMemcpy(dev_imgBuf, imgPtr, imgBufSize * sizeof(Pixel), cudaMemcpyHostToDevice));

		// Generate distribution
		if (params.timing)
			CUDA_CALL(cudaEventRecord(start, 0));
		computePixelIntensities<<<blockSize, threadSize>>>(dev_imgBuf, dev_intensities, dims);
		generateImagePyramid<<<blockSize, threadSize>>>(dev_intensities, dev_imgPyramid, dims);
		detectEdges<<<blockSize, threadSize>>>(dev_imgPyramid, dev_edges, dims);
		blendDistributionData<<<blockSize, threadSize>>>(dev_intensities, dev_edges,
			dev_pointHistoricity, dev_distr, dims, params.intensityEdgeWeight,
			params.historicityWeight);
		if (params.debug)
			visualizeDistr(dev_distr, dims);
        generateCDF<<<blockSize, threadSize>>>(dev_distr, dims.x*dims.y, dims);
		if (params.timing)
			CUDA_CALL(cudaEventRecord(distrGenerated, 0));

        // Sample n points from the distribution
		samplePoints<<<blockSize, threadSize>>>(dev_distr, imgBufSize, dev_pointBuf, numPoints,
			dev_pointHistoricity, dims);
		if (params.timing)
			CUDA_CALL(cudaEventRecord(pointsSampled, 0));

        // Perform tesselation of points (with one or more kernels)
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
	CUDA_CALL(cudaFree(dev_imgPyramid));
	CUDA_CALL(cudaFree(dev_edges));
	CUDA_CALL(cudaFree(dev_pointHistoricity));
    CUDA_CALL(cudaFree(dev_distr));
	CUDA_CALL(cudaFree(dev_pointBuf));

    return NOT_YET_IMPLEMENTED;
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


__global__ void generateImagePyramid(ChannelBuf intensities, ChannelBuf pyramid, short2 dims)
{
	// Define a 3x3 Gaussian filter with which the per-layer blurring will be performed
	const float gaussianFilter[3][3] =
	{
		{ 0.0625f, 0.1250f, 0.0625f },
		{ 0.1250f, 0.2500f, 0.1250f },
		{ 0.0625f, 0.1250f, 0.0625f },
	};

	// The x and y coordinates for which this instance of the kernel is responsible
	short x = threadIdx.x + blockIdx.x * blockDim.x;
	short y = threadIdx.y + blockIdx.y * blockDim.y;
	uint32_t offset = x + (y * dims.x);
	if (x > dims.x || y > dims.y)
		return;

	// Copy in grayscale value
	pyramid[offset] = intensities[offset];

	__syncthreads();

	// Initialize the required offsets
	uint32_t pyramidLevelOffset = 0;
	uint8_t numPyramidLevels = 0;
	float val;

	// Throw out every other row and column per iteration
	while (x % 2 == 0 && y % 2 == 0
		&& dims.x > MIN_PYRAMID_SIZE && dims.y > MIN_PYRAMID_SIZE
		&& numPyramidLevels < MAX_PYRAMID_LEVELS)
	{
		val = 0;
		for (int i = offset - dims.x, ii = 0; i <= offset + dims.x; i += dims.x, ++ii)
		{
			for (int j = i - 1, jj = 0; j <= i + 1; ++j, ++jj)
			{
				// If selected pixel is within the bounds, filter it; otherwise, do nothing
				if (j >= pyramidLevelOffset && j < pyramidLevelOffset + (dims.x * dims.y))
				{
					val += gaussianFilter[ii][jj] * pyramid[j];
				}
			}
		}

		// Update the bounds of the image for the next iteration
		pyramidLevelOffset += dims.x * dims.y;
		dims.x /= 2;
		dims.y /= 2;
		x /= 2;
		y /= 2;

		offset = pyramidLevelOffset + x + (y * dims.x);
		++numPyramidLevels;

		__syncthreads();

		pyramid[offset] = static_cast<channel_t>(val);

		__syncthreads();
	}
}


__global__ void detectEdges(ChannelBuf imgPyramid, ChannelBuf edgesPyramid, short2 dims)
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
	if (x > dims.x || y > dims.y)
		return;

	// Initialize the starting offsets
	unsigned int pyramidLevelOffset = 0;
	unsigned int offset = x + (y * dims.x);
	uint8_t numPyramidLevels = 0;

	float val;
	while (x < dims.x && y < dims.y && dims.x > MIN_PYRAMID_SIZE && dims.y > MIN_PYRAMID_SIZE
		&& numPyramidLevels < MAX_PYRAMID_LEVELS)
	{
		// Calculate the image gradient value at this location for this pyramid level
		val = 0;
		for (int i = offset - dims.x, ii = 0; i <= offset + dims.x; i += dims.x, ++ii)
		{
			for (int j = i - 1, jj = 0; j <= i + 1; ++j, ++jj)
			{
				if (j >= pyramidLevelOffset && j < pyramidLevelOffset + (dims.x * dims.y))
				{
					val += edgeFilter[ii][jj] * imgPyramid[j];
				}
			}
		}

		// Save the calculated edge intensity with light filtering
		edgesPyramid[offset] = (channel_t) abs(val) > EDGE_THRESH ? abs(val) : 0;

		// Update the offset for the next pyramid level
		pyramidLevelOffset += dims.x * dims.y;
		dims.x /= 2;
		dims.y /= 2;
		offset = pyramidLevelOffset + x + (y * dims.x);
		++numPyramidLevels;
	}
}


__global__ void blendDistributionData(ChannelBuf intensity, ChannelBuf edges,
	ChannelBuf historicity, double* distribution, short2 dims, float intEdgeWeight,
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


__global__ void generateCDF(double* distribution, size_t distrLen, short2 dims)
{
    // The x and y coordinates for which this instance of the kernel is responsible
    short x = threadIdx.x + blockIdx.x * blockDim.x;
    short y = threadIdx.y + blockIdx.y * blockDim.y;
	uint32_t offset = x + (y * dims.x);
    if (offset > distrLen)
        return;

    // Calculate cumulative sum
    //      http://http.developer.nvidia.com/GPUGems3/gpugems3_ch39.html

    // Normalize the distribution
    // distribution[offset] /= distribution[distrLen - 1];
}


__global__ void samplePoints(double* distribution, size_t distrLen, short2* pointBuf,
	uint32_t numPoints, ChannelBuf historicityBuf, short2 dims)
{
    // The x and y coordinates for which this instance of the kernel is responsible
    short x = threadIdx.x + blockIdx.x * blockDim.x;
    short y = threadIdx.y + blockIdx.y * blockDim.y;
	uint32_t offset = x + (y * dims.x);
    if (x > dims.x || y > dims.y)
        return;

    // TODO Generate random numbers and index into distribution
    // TODO Convert distribution index into point
    // TODO Store points into blue channel of dev_distrCalcBuf
}