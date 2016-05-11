/*
 * imageProcessor.hpp
 *
 * Author: Robert Miller
 * Last Edit: 5/10/16
 *
 * The heart of the application's rendering pipeline, the processImageResource function (and its
 *   related subroutines) are massive enough to merit their own file--and future refactoring might
 *   put the code into multiple independent files.
 */


#pragma once


#include "imageResource.hpp"
#include "types.hpp"


const uint16_t THREADS_PER_BLOCK = 32;
const uint16_t EDGE_THRESH = 10;


ReturnCode processImageResource(ImageResource& input, ImageResource& output, ParamBundle params);