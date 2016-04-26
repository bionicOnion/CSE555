/*
 * constants.h
 * Author: Robert Miller
 * Last Edited: 2/2/16
 *
 * Defines the constant values used throughout the application
 */


#pragma once

#include <string>

#include <cuda_runtime.h>


// Argument processing
const unsigned short PNAME_ARG_INDEX = 0;
const unsigned short IMG_ARG_INDEX = 1;
const unsigned short ALIGN_MODE_INDEX = 2;
const unsigned short X_WINDOW_RANGE = 3;
const unsigned short Y_WINDOW_RANGE = 4;
const unsigned short EXPECTED_NUM_ARGS = 3;
const unsigned short EXPECTED_NUM_ARGS_WINDOW = 5;
const std::string MULTI_LAYER_SPECIFIER = "-m";
const std::string SINGLE_LAYER_SPECIFIER = "-s";

// Return codes
const short SUCCESS = 0;
const short INCORRECT_USAGE = -1;
const short DISCONT_MATRIX = -2;
const short DEV_ALLOC_FAIL = -3;
const short DEV_CPY_FAIL = -4;
const short GPU_TIMING_FAIL = -5;
const short HOST_CPY_FAIL = -6;

// Channel alignment
const float BORDER_CUT_MARGIN = 0.075;
const short NUM_ALIGN_LEVELS = 6;
const short NUM_ALIGN_NEIGHBORS = 9;

// Color constants (OpenCV uses a BGR color system)
const short COLOR_MAX = 255;
const short COLOR_MIN = 0;
const short NUM_CHANNELS = 3;
const short RED = 2;
const short GREEN = 1;
const short BLUE = 0;
__constant__ float RED_MAPPING[3]   ={ 0.0, 0.0, 1.0 };
__constant__ float GREEN_MAPPING[3] ={ 0.0, 1.2, 0.0 };
__constant__ float BLUE_MAPPING[3]  ={ 0.8, 0.0, 0.0 };

const short EDGE_THRESH = 48;
const short MIN_PYRAMID_SIZE = 64;
const short MAX_SMALL_IMG_DIM = 512;
const short THREADS_PER_BLOCK = 16;