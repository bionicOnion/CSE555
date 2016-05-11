/*
 * util.cpp
 *
 * Author: Robert Miller
 * Last Edit: 5/10/16
 *
 * Implementations of the utility functions defined in util.hpp
 */


#include <iostream>

// OpenCV
#include <opencv2/opencv.hpp>

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
    case GL_ERROR:
        std::cout << "An error occured with OpenGL"
            << " (code " << errCode << ")" << std::endl;
        break;
    case GL_COMPILE_ERROR:
        std::cout << "An OpenGL shader failed to compile"
            << " (code " << errCode << ")" << std::endl;
        break;
    case GL_LINK_ERROR:
        std::cout << "OpenGL shader program failed to link"
            << " (code " << errCode << ")" << std::endl;
        break;
    case GL_VALIDATE_ERROR:
        std::cout << "OpenGL shader program failed to validate"
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
        << errCode << " (" << cudaGetErrorString(errCode) << ") " << std::endl;

    return CUDA_ERROR;
}


ReturnCode printErrorMsgGL(GLenum errCode, std::string file, int lineNum)
{
    if (errCode == GL_NO_ERROR)
        return SUCCESS;

    std::cout << "In file " << file << " at line " << lineNum << ": An OpenGL call failed with code "
        << errCode << " (" << gluErrorString(errCode) << ") " << std::endl;

    return GL_ERROR;
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


void visualizeDistr(float* distr, short2 dims)
{
    auto buf = reinterpret_cast<float*>(malloc(dims.x * dims.y * sizeof(float)));
    if (cudaMemcpy(buf, distr, dims.x*dims.y*sizeof(float), cudaMemcpyDeviceToHost) != cudaSuccess)
    {
        free(buf);
        return;
    }
    imshow("Distribution", cv::Mat(dims.y, dims.x, CV_32FC1, buf));
    cv::waitKey();
    free(buf);
}


ReturnCode printTimings(cudaEvent_t start, cudaEvent_t distrGenerated, cudaEvent_t pointsSampled,
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

    return SUCCESS;
}


void printDeviceData(cudaDeviceProp prop)
{
    // Determine the number of CUDA cores in the selected device
    uint16_t coreCount;
    switch (prop.major)
    {
    case 1:
        coreCount = prop.multiProcessorCount * 8;
        break;
    case 2:
        if (prop.minor == 0)
            coreCount = prop.multiProcessorCount * 32;
        else
            coreCount = prop.multiProcessorCount * 48;
        break;
    case 3:
    case 4:
        coreCount = prop.multiProcessorCount * 192;
        break;
    case 5:
    case 6:
    case 7:
        coreCount = prop.multiProcessorCount * 128;
        break;
    default:
        coreCount = 0;
        break;
    }

    std::cout << "Selected CUDA device properties:" << std::endl;
    std::cout << "  Device name:    " << prop.name << std::endl;
    std::cout << "  Compute level:  " << prop.major << '.' << prop.minor << std::endl;
    std::cout << "  CUDA cores:     " << coreCount << std::endl;
    std::cout << "  Clock rate:     " << (prop.clockRate >> 10) << " MHz" << std::endl;
    std::cout << "  Available VRAM: " << (prop.totalGlobalMem >> 30) << " GB" << std::endl;
    std::cout << std::endl;
}