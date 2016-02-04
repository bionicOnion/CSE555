/*
 * accelerator.cuh
 * Author: Robert Miller
 * Last Edited: 2/2/16
 *
 * Defines the prototypes of the CUDA kernels invoked from main.cu to perform hardware- accelarated composition of
 *   three-channel images. Each of these functions executes on the GPU in a massively parallel context; values are
 *   computed per-pixel and stored into shared buffers, greatly improving the speed over a CPU implementation (which
 *   would necessitate a 2D loop or an equivalent) to calculate the same values.
 */


#pragma once

#include <stdint.h>


 typedef uint8_t Pixel;
 typedef Pixel* Image;


 __global__ void generateImagePyramids(Image red, Image green, Image blue, short2 imgDims);
__global__ void detectEdges(Image red, Image green, Image blue, Image redEdges, Image greenEdges,
	Image blueEdges, short2 imgDims);
__global__ void alignImages(Image baseEdges, Image alignEdges, short2 imgDims, short2* alignment,
	unsigned long long* errSumBuf);
__global__ void produceComposite(Image red, Image green, Image blue, Image composite,
	short2* grOffset, short2* gbOffset, short2 imgDims);