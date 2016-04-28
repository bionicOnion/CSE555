/*
 * TODO
 */


// CUDA
#include <cuda_runtime.h>

#include "imageProcessor.hpp"
#include "util.hpp"


#define CUDA_CALL(CALL) if ((cudaRetCode = (CALL)) != cudaSuccess) PRINT_ERR_MSG_CUDA(cudaRetCode)


// TODO add debug #ifdefs to look at distribution, Voronoi, etc.
ReturnCode processImageResource(ImageResource& input, ImageResource& output, ParamBundle params)
{
	cudaError_t cudaRetCode;

    // Prepare relevant state
    auto dims = make_short2(input.getWidth(), input.getHeight());
    uint32_t imgBufSize = dims.x * dims.y;

    // Allocate memory
    auto host_imgBuf = reinterpret_cast<Image>(malloc(imgBufSize * sizeof(Pixel)));

    Image dev_imgBuf;
    CUDA_CALL(cudaMalloc(&dev_imgBuf, imgBufSize * sizeof(Pixel)));

    // Perform processing on the image/video
    for (auto i = 0; i < input.getFrameCount(); ++i)
    {
        // Copy input frame to the GPU
		auto imgPtr = input.getFrame(i);
		if (imgPtr == nullptr)
			break;
		CUDA_CALL(cudaMemcpy(dev_imgBuf, imgPtr, imgBufSize * sizeof(Pixel), cudaMemcpyHostToDevice));

        // Call kernel(s) to generate probability distribution
		//   Compute pixel intensities
		//   Generate image pyramid and detect edges at each layer
		//   Blend layers together to form single edge image
		//   Blend intensities with edges according to params.intensityEdgeWeight
		//   Blend distribution with historical point positions according to params.historicityWeight
		//   Compute cumulative sum/normalize the distribution
        // Call kernel to sample points from the generated distribution
		//   Number of points sampled = dims.x * dims.y * params.pointRatio
        // Perform tesselation of points (with one or more kernels)
		//   Follow the paper linked to in the project proposal
        // Call out to OpenGL to generate the final image; store it in devImg

        // Copy the reulting frame back to the CPU
        CUDA_CALL(cudaMemcpy(host_imgBuf, dev_imgBuf, imgBufSize * sizeof(Pixel), cudaMemcpyDeviceToHost));
        output.addFrame(host_imgBuf, dims.x, dims.y);
    }

    // Free memory
    free(host_imgBuf);
    CUDA_CALL(cudaFree(dev_imgBuf));

    return NOT_YET_IMPLEMENTED;
}