/*
 * TODO
 */


#include <iostream>

// OpenCV
#include <opencv2/opencv.hpp>

#include "constants.hpp"
#include "types.hpp"
#include "util.hpp"


ReturnCode printErrorMsg(ReturnCode errCode, std::string file, int lineNum)
{
    if (errCode == SUCCESS)
        return SUCCESS;

    std::cout << "In file " << file << " at line " << lineNum << ": ";

    switch(errCode)
    {
    case ARG_OUT_OF_BOUNDS:
        std::cout << "A provided argument was out of the allowed bounds"
            << " (code " << errCode << ")" << std::endl;
        break;
    case INSUFFICIENT_ARGS:
        std::cout << "Too few arguments were provided for the selected options"
            << " (code " << errCode << ")" << std::endl;
        break;
    case INVALID_ARGUMENT:
        std::cout << "A provided argument was not legal in the context in which it appeared"
            << " (code " << errCode << ")" << std::endl;
        break;
    case TOO_MANY_INPUT_FILES:
        std::cout << "More than one input file was specified"
            << " (code " << errCode << ")" << std::endl;
        break;
    case UNRECOGNIZED_ARG:
        std::cout << "An argument code was not recognized"
            << " (code " << errCode << ")" << std::endl;
        break;
    case UNRECOGNIZED_COLORING_MODE:
        std::cout << "A coloring mode was not recognized"
            << " (code " << errCode << ")" << std::endl;
        break;
    case USAGE_PRINTED:
        break;
    case NO_INPUT_FILE:
        std::cout << "No input file was specified"
            << " (code " << errCode << ")" << std::endl;
        break;
    case FILE_NOT_OPENED:
        std::cout << "The specified input file could not be opened"
            << " (code " << errCode << ")" << std::endl;
        break;
    case FILE_WRITE_ERROR:
        std::cout << "Writing to the specified output file failed"
            << " (code " << errCode << ")" << std::endl;
        break;
    case RESOURCE_UNINITIALIZED:
        std::cout << "No data has been initialized for an ImageResource"
            << " (code " << errCode << ")" << std::endl;
        break;
    case CUDA_ERROR:
        std::cout << "A CUDA error occured"
            << " (code " << errCode << ")" << std::endl;
        break;
    case NOT_YET_IMPLEMENTED:
        std::cout << "A function has not yet been implemented"
            << " (code " << errCode << ")" << std::endl;
        break;
    case UNRECOGNIZED_INPUT_TYPE:
        std::cout << "The input type specified was not recognized"
            << " (code " << errCode << ")" << std::endl;
        break;
    default:
        std::cout << "The error code " << errCode << " was not recognized." << std::endl;
        break;
    }

    return NOT_YET_IMPLEMENTED; // TODO return errCode;
}


ReturnCode printErrorMsgCUDA(cudaError_t errCode, std::string file, int lineNum)
{
    if (errCode == cudaSuccess)
        return SUCCESS;

	std::cout << "In file " << file << " at line " << lineNum << ": A CUDA call failed with code "
		<< errCode << "(" << cudaGetErrorString(errCode) << ")" << std::endl;

    return CUDA_ERROR;
}


void displayPreviewImage(Image img, short2 dims)
{
    imshow("Image Preview", cv::Mat(dims.y, dims.x, CV_8UC3, img));
    cv::waitKey();
}


void displayPreviewImageGPU(Image img, short2 dims)
{
    auto buf = reinterpret_cast<Image>(malloc(dims.x * dims.y * sizeof(Pixel)));
    if (cudaMemcpy(buf, img, dims.x*dims.y*sizeof(Pixel), cudaMemcpyDeviceToHost) != cudaSuccess)
    {
        free(buf);
        return;
    }

    imshow("Image Preview", cv::Mat(dims.y, dims.x, CV_8UC3, buf));
    cv::waitKey();
    free(buf);
}


void displayPreviewChannelBufGPU(ChannelBuf img, short2 dims)
{
	auto buf = reinterpret_cast<Image>(malloc(dims.x * dims.y * sizeof(channel_t)));
	if (cudaMemcpy(buf, img, dims.x*dims.y*sizeof(channel_t), cudaMemcpyDeviceToHost) != cudaSuccess)
	{
		free(buf);
		return;
	}

	imshow("Image Preview", cv::Mat(dims.y, dims.x, CV_8UC1, buf));
	cv::waitKey();
	free(buf);
}


void displayImagePyramidGPU(ChannelBuf img, short2 dims)
{
	uint32_t pyramidOffset = 0;
	uint8_t numPyramidLevels = 0;
	while (dims.x > MIN_PYRAMID_SIZE && dims.y > MIN_PYRAMID_SIZE
		&& numPyramidLevels < MAX_PYRAMID_LEVELS)
	{
		displayPreviewChannelBufGPU(img + pyramidOffset, dims);
		pyramidOffset += dims.x * dims.y;
		dims.x /= 2;
		dims.y /= 2;
		++numPyramidLevels;
	}
}


void visualizeDistr(double* distr, short2 dims)
{
    auto buf = reinterpret_cast<double*>(malloc(dims.x * dims.y * sizeof(double)));
	if (cudaMemcpy(buf, distr, dims.x*dims.y*sizeof(double), cudaMemcpyDeviceToHost) != cudaSuccess)
    {
        free(buf);
        return;
    }
    imshow("Distribution", cv::Mat(dims.y, dims.x, CV_64FC1, buf));
    cv::waitKey();
    free(buf);
}


void printTimings(cudaEvent_t start, cudaEvent_t distrGenerated, cudaEvent_t pointsSampled,
	cudaEvent_t pointsTesselated, cudaEvent_t end, int frameIdx)
{
	cudaError_t cudaRetCode;

	// Calculate elapsed amount of time
	float distrTime, samplingTime, tesselTime, drawTime, totalTime;
	CUDA_CALL(cudaEventElapsedTime(&distrTime, start, distrGenerated));
	CUDA_CALL(cudaEventElapsedTime(&samplingTime, distrGenerated, pointsSampled));
	CUDA_CALL(cudaEventElapsedTime(&tesselTime, pointsSampled, pointsTesselated));
	CUDA_CALL(cudaEventElapsedTime(&drawTime, pointsTesselated, end));
	CUDA_CALL(cudaEventElapsedTime(&totalTime, start, end));

	std::cout << "Timings for frame " << frameIdx << ':' << std::endl;
	std::cout << "  Distribution generation: " << distrTime << " ms" << std::endl;
	std::cout << "  Point sampling:          " << samplingTime << " ms" << std::endl;
	std::cout << "  Tesselation:             " << tesselTime << " ms" << std::endl;
	std::cout << "  Drawing:                 " << drawTime << " ms" << std::endl;
	std::cout << "  Total GPU time:          " << totalTime << " ms" << std::endl;
	std::cout << std::endl;
}