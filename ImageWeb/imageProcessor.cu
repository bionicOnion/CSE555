/*
 * TODO
 */


#include <ctime>

// CUDA
#include <cuda_runtime.h>
#include <curand_kernel.h>

#include "constants.hpp"
#include "imageProcessor.hpp"
#include "util.hpp"


// Forward declarations
__global__ void cuRAND_init(curandState_t* states, uint64_t seed, short2 dims);
__global__ void computePixelIntensities(Image img, ChannelBuf intensities, short2 dims);
__global__ void detectEdges(ChannelBuf intensities, ChannelBuf edges, short2 dims);
__global__ void blendDistributionData(ChannelBuf intensity, ChannelBuf edges,
    ChannelBuf historicity, double* distribution, short2 dims, float intEdgeWeight,
    float historicityWeight);
__global__ void generateCDF(double* distribution, size_t distrLen, short2 dims);
__global__ void samplePoints(double* distribution, size_t distrLen, curandState_t* randStates,
    short2* pointBuf, uint32_t numPoints, ChannelBuf historicityBuf, short2 dims);


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
    ChannelBuf dev_intensities, dev_edges, dev_pointHistoricity;
    double* dev_distr;
    short2* dev_pointBuf;
    curandState_t* dev_curandStates;
    CUDA_CALL(cudaMalloc(&dev_imgBuf, imgBufSize * sizeof(Pixel)));
    CUDA_CALL(cudaMalloc(&dev_intensities, imgBufSize * sizeof(channel_t)));
    CUDA_CALL(cudaMalloc(&dev_edges, imgBufSize * sizeof(channel_t)));
    CUDA_CALL(cudaMalloc(&dev_pointHistoricity, imgBufSize * sizeof(channel_t)));
    CUDA_CALL(cudaMalloc(&dev_distr, imgBufSize * sizeof(double)));
    CUDA_CALL(cudaMalloc(&dev_pointBuf, numPoints * sizeof(short2)));
    CUDA_CALL(cudaMalloc(&dev_curandStates, imgBufSize * sizeof(curandState_t)));
    auto host_imgBuf = reinterpret_cast<Image>(malloc(imgBufSize * sizeof(Pixel)));

    CUDA_CALL(cudaMemset(dev_pointHistoricity, 0, imgBufSize * sizeof(channel_t)));

    // Initialize state for cuRAND
    cuRAND_init<<<blockSize, threadSize>>>(dev_curandStates, params.debug ? 0 : time(NULL), dims);

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
        detectEdges<<<blockSize, threadSize>>>(dev_intensities, dev_edges, dims);
        blendDistributionData<<<blockSize, threadSize>>>(dev_intensities, dev_edges,
            dev_pointHistoricity, dev_distr, dims, params.intensityEdgeWeight,
            params.historicityWeight);
        if (params.debug)
            visualizeDistr(dev_distr, dims);
        generateCDF<<<blockSize, threadSize>>>(dev_distr, dims.x*dims.y, dims);
        if (params.timing)
            CUDA_CALL(cudaEventRecord(distrGenerated, 0));

        // Sample n points from the distribution
        samplePoints<<<blockSize, threadSize>>>(dev_distr, imgBufSize, dev_curandStates,
            dev_pointBuf, numPoints, dev_pointHistoricity, dims);
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
    CUDA_CALL(cudaFree(dev_edges));
    CUDA_CALL(cudaFree(dev_pointHistoricity));
    CUDA_CALL(cudaFree(dev_distr));
    CUDA_CALL(cudaFree(dev_pointBuf));
    CUDA_CALL(cudaFree(dev_curandStates));

    return NOT_YET_IMPLEMENTED;
}


__global__ void cuRAND_init(curandState_t* states, uint64_t seed, short2 dims)
{
    // The x and y coordinates for which this instance of the kernel is responsible
    short x = threadIdx.x + blockIdx.x * blockDim.x;
    short y = threadIdx.y + blockIdx.y * blockDim.y;
    uint32_t offset = x + (y * dims.x);
    if (x > dims.x || y > dims.y)
        return;

    // curand_init(seed, offset, 0, &states[offset]);
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


__global__ void samplePoints(double* distribution, size_t distrLen, curandState_t* randStates,
    short2* pointBuf, uint32_t numPoints, ChannelBuf historicityBuf, short2 dims)
{
    // The x and y coordinates for which this instance of the kernel is responsible
    short x = threadIdx.x + blockIdx.x * blockDim.x;
    short y = threadIdx.y + blockIdx.y * blockDim.y;
    uint32_t offset = x + (y * dims.x);
    if (x > dims.x || y > dims.y)
        return;

    // Select a random pixel location according to the provide distribution
    //   Once a random value is samples, binary search is employed to find the corresponding point
    // auto randVal = curand_uniform_double(&randStates[offset]);

    // TODO Store points into historicityBuf
}