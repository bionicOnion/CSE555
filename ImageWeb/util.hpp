/*
 * TODO
 */


#pragma once

// CUDA
#include <cuda_runtime.h>

#include "types.hpp"


ReturnCode printErrorMsg(ReturnCode errCode, std::string file, int lineNum);
ReturnCode printErrorMsgCUDA(cudaError_t errCode, std::string file, int lineNum);


#define PRINT_ERR_MSG(ERR_CODE) printErrorMsg(ERR_CODE, __FILE__, __LINE__)
#define PRINT_ERR_MSG_CUDA(ERR_CODE) printErrorMsgCUDA(ERR_CODE, __FILE__, __LINE__)