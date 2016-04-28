/*
 * TODO
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
        << errCode << "(" << cudaGetErrorString(errCode) << ")";

    return CUDA_ERROR;
}


void displayPreviewImage(Image img, short2 dims)
{
    imshow("Image Preview", cv::Mat(dims.y, dims.x, CV_8UC3, img));
    cv::waitKey();
}