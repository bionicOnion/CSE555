/*
 * util.hpp
 *
 * Author: Robert Miller
 * Last Edit: 5/10/16
 *
 * An assortment of utility functions primarily useful for debugging or (int the case of macros)
 *   simplifying error handling and reporting
 */


#pragma once

// CUDA
#include <cuda_runtime.h>

// OpenGL
#include <GL\glew.h>

#include "types.hpp"


// Error reporting
ReturnCode printErrorMsg(ReturnCode errCode, std::string file, int lineNum);
ReturnCode printErrorMsgCUDA(cudaError_t errCode, std::string file, int lineNum);
ReturnCode printErrorMsgGL(GLenum errCode, std::string file, int lineNum);

// Data buffer/image visualization
void displayPreviewImage(Image img, short2 dims);
void displayPreviewImageGPU(Image img, short2 dims);
void displayPreviewChannelBufGPU(ChannelBuf img, short2 dims);
void visualizeDistr(float* distr, short2 dims);

// Print timing info
ReturnCode printTimings(cudaEvent_t start, cudaEvent_t distrGenerated, cudaEvent_t pointsSampled,
	cudaEvent_t pointsTesselated, cudaEvent_t end, int frameIdx);

// Print system info
void printDeviceData(cudaDeviceProp prop);


#define PRINT_ERR_MSG(ERR_CODE) printErrorMsg(ERR_CODE, __FILE__, __LINE__)
#define PRINT_ERR_MSG_CUDA(ERR_CODE) printErrorMsgCUDA(ERR_CODE, __FILE__, __LINE__)
#define PRINT_ERR_MSG_GL(ERR_CODE) printErrorMsgGL(ERR_CODE, __FILE__, __LINE__)


#define CUDA_CALL(CALL) if ((cudaRetCode = (CALL)) != cudaSuccess) return PRINT_ERR_MSG_CUDA(cudaRetCode)