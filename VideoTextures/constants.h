/*
 * constants.h
 */

#pragma once


#include <stdint.h>


#define min(a, b) ((a) < (b) ? (a) : (b))


// Argument constants
const unsigned short MIN_ARG_COUNT = 3;
const unsigned short PNAME_INDEX = 0;
const unsigned short TEX_VID_INDEX = 1;
const unsigned short GEN_VID_LEN_INDEX = 2;

// Return codes
const int SUCCESS = 0;
const int INSUFFICIENT_ARGS = 1;
const int FILE_NOT_OPENED = 2;
const int IDENTICAL_IMAGES = 3;

// Progress bar constants
const uint8_t DEFAULT_PBAR_WIDTH = 64;
const unsigned int PBAR_TITLE_WIDTH = 26;
const uint8_t PERCENT_WIDTH = 10;

// General constants
const uint16_t MAX_FRAMES = 20;
const float SIGMA_FACTOR = 0.25f;
const float VID_SCALE_FACTOR = 0.25f;
const float WEIGHTS[] = { 1.0f/16, 4.0f/16, 6.0f/16, 4.0f/16, 1.0f/16 };
const int WEIGHT_TAP_COUNT = 2;