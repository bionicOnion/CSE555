/*
 * constants.cpp
 *
 * Author: Robert Miller
 * Last Edit: 5/10/16
 *
 * Defines a variety of compile-time constants used throughout the application
 */


#include <cstdint>


const uint16_t THREADS_PER_BLOCK = 32;

const uint16_t MIN_PYRAMID_SIZE = 64;
const uint8_t MAX_PYRAMID_LEVELS = 5;

const short EDGE_THRESH = 10;

const short MAX_DIM = 640;

const uint8_t DEFAULT_PBAR_WIDTH = 64;
const uint8_t PERCENT_WIDTH = 10;