/*
 * TODO
 */


// CUDA
#include <cuda_runtime.h>

#include "imageProcessor.hpp"
#include "util.hpp"


#define CUDA_CALL(CALL) if ((cudaRetCode = (CALL)) != cudaSuccess) PRINT_ERR_MSG_CUDA(cudaRetCode)


// CUDA constants
const uint16_t THREADS_PER_BLOCK = 16;


// Forward declarations
__global__ void computePixelIntensities(Image img, Image distrCalcBuf, short2 dims);
__global__ void computeEdgeData(Image distrCalcBuf, short2 dims);
__global__ void blendDistributionData(Image distrCalcBuf, double* distribution, short2 dims,
    float intEdgeWeight, float historicityWeight);
__global__ void generateCDF(double* distribution, size_t distrLen);
__global__ void samplePoints(double* distribution, size_t distrLen, short2* pointBuf,
    uint32_t numPoints, Image distrCalcBuf, short2 dims);


// TODO add debug #ifdefs to look at distributiondouble*, Voronoi, etc.
ReturnCode processImageResource(ImageResource& input, ImageResource& output, ParamBundle params)
{
	cudaError_t cudaRetCode;

    // Prepare relevant state
    auto dims = make_short2(input.getWidth(), input.getHeight());
    uint32_t imgBufSize = dims.x * dims.y;
    uint32_t numPoints = imgBufSize * params.pointRatio;
    dim3 blockSize(dims.x / THREADS_PER_BLOCK, dims.y / THREADS_PER_BLOCK);
    dim3 threadSize(THREADS_PER_BLOCK, THREADS_PER_BLOCK);

    // Allocate memory
    auto host_imgBuf = reinterpret_cast<Image>(malloc(imgBufSize * sizeof(Pixel)));

    Image dev_imgBuf, dev_distrCalcBuf;
    double* dev_distr;
    short2* pointBuf;
    CUDA_CALL(cudaMalloc(&dev_imgBuf, imgBufSize * sizeof(Pixel)));
    CUDA_CALL(cudaMalloc(&dev_distrCalcBuf, imgBufSize * sizeof(Pixel) * 4 / 3));
    CUDA_CALL(cudaMalloc(&dev_distr, imgBufSize * sizeof(double)));
    CUDA_CALL(cudaMalloc(&pointBuf, numPoints * sizeof(short2)));

    CUDA_CALL(cudaMemset(dev_distrCalcBuf, 0, imgBufSize * sizeof(Pixel) * 4 / 3));

    // Perform processing on the image/video
    // TODO insert timings for benchmarking
    for (auto i = 0; i < input.getFrameCount(); ++i)
    {
        // Copy input frame to the GPU
		auto imgPtr = input.getFrame(i);
		if (imgPtr == nullptr)
			break;
		CUDA_CALL(cudaMemcpy(dev_imgBuf, imgPtr, imgBufSize * sizeof(Pixel), cudaMemcpyHostToDevice));

		// Generate distribution
        computePixelIntensities<<<blockSize, threadSize>>>(dev_imgBuf, dev_intensityBuf, dims);
        computeEdgeData<<<blockSize, threadSize>>>(dev_intensityBuf, dev_edgeBuf, dims);
        blendDistributionData<<<blockSize, threadSize>>>(dev_intensityBuf, dev_edgeBuf,
            dev_pointLocBuf, dev_distr, dims, params.intensityEdgeWeight, params.historicityWeight);
        // TODO copy distribution back to host for debugging visualitzation
        generateCDF<<<blockSize, threadSize>>>(dev_distr, dims);

        // Sample n points from the distribution
        samplePoints<<<blockSize, threadSize>>>(dev_distr, imgBufSize, pointBuf, numPoints);

        // Perform tesselation of points (with one or more kernels)
		//   Follow the paper linked to in the project proposal

        // Call out to OpenGL to generate the final image; store it in dev_imgBuf
        //   TODO

        // Copy the reulting frame back to the CPU
        CUDA_CALL(cudaMemcpy(host_imgBuf, dev_imgBuf, imgBufSize * sizeof(Pixel), cudaMemcpyDeviceToHost));
        output.addFrame(host_imgBuf, dims.x, dims.y);
    }

    // Free memory
    free(host_imgBuf);
    CUDA_CALL(cudaFree(dev_imgBuf));
    CUDA_CALL(cudaFree(dev_intensityBuf));
    CUDA_CALL(cudaFree(dev_edgeBuf));
    CUDA_CALL(cudaFree(dev_pointLocBuf));
    CUDA_CALL(cudaFree(dev_distr));

    return NOT_YET_IMPLEMENTED;
}


__global__ void computePixelIntensities(Image img, Image distrCalcBuf, short2 dims)
{
    // The x and y coordinates for which this instance of the kernel is responsible
    short x = threadIdx.x + blockIdx.x * blockDim.x;
    short y = threadIdx.y + blockIdx.y * blockDim.y;
    uint16_t offset = x + (y * dims.x);
    if (x > dims.x || y > dims.y)
        return;

    // Store intensity data in the red channel of the 
    distrCalcBuf[offset].r = (img[offset].r + img[offset].g + img[offset].b) / 3;
}


__global__ void computeEdgeData(Image distrCalcBuf, short2 dims)
{
    // The x and y coordinates for which this instance of the kernel is responsible
    short x = threadIdx.x + blockIdx.x * blockDim.x;
    short y = threadIdx.y + blockIdx.y * blockDim.y;
    uint16_t offset = x + (y * dims.x);
    if (x > dims.x || y > dims.y)
        return;

    // TODO Store edge data in green channel
}


__global__ void blendDistributionData(Image distrCalcBuf, double* distribution, short2 dims,
    float intEdgeWeight, float historicityWeight)
{
    // The x and y coordinates for which this instance of the kernel is responsible
    short x = threadIdx.x + blockIdx.x * blockDim.x;
    short y = threadIdx.y + blockIdx.y * blockDim.y;
    uint16_t offset = x + (y * dims.x);
    if (x > dims.x || y > dims.y)
        return;

    // Calculate the value for this point in the distribution
    distribution[offset] = intEdgeWeight*distrCalcBuf[offset].r
        + (1 - intEdgeWeight)*distrCalcBuf[offset].g;
    distribution[offset] = historicityWeight*distrCalcBuf[offset].b
        + (1 - historicityWeight)*distribution[offset];
}


__global__ void generateCDF(double* distribution, size_t distrLen)
{
    // The x and y coordinates for which this instance of the kernel is responsible
    short x = threadIdx.x + blockIdx.x * blockDim.x;
    short y = threadIdx.y + blockIdx.y * blockDim.y;
    uint16_t offset = x + (y * dims.x);
    if (x > dims.x || y > dims.y)
        return;

    // Calculate cumulative sum
    // TODO

    __syncthreads();

    // Normalize the distribution
    distribution[offset] /= distribution[distrLen - 1];
}


__global__ void samplePoints(double* distribution, size_t distrLen, short2* pointBuf,
    uint32_t numPoints, Image distrCalcBuf, short2 dims)
{
    // The x and y coordinates for which this instance of the kernel is responsible
    short x = threadIdx.x + blockIdx.x * blockDim.x;
    short y = threadIdx.y + blockIdx.y * blockDim.y;
    uint16_t offset = x + (y * dims.x);
    if (x > dims.x || y > dims.y)
        return;

    // TODO Generate random numbers and index into distribution
    // TODO Convert distribution index into point
    // TODO Store points into blue channel of dev_distrCalcBuf
}