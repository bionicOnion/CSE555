/*
 * accelerator.cu
 * Author: Robert Miller
 * Last Edited: 2/2/16
 *
 * Implementation of the kernelized image composition system defined in accelerator.cuh
 */


#include <math.h>

// CUDA
#include <cuda_runtime.h>

#include "accelerator.cuh"
#include "constants.cuh"


__device__ inline float clampFloat(float val, float min, float max)
	{ return val > min ? (val < max ? val : max) : min; }


__device__ inline short clampShort(short val, short min, short max)
	{ return val > min ? (val < max ? val : max) : min; }


__global__ void generateImagePyramids(Image red, Image green, Image blue, short2 imgDims)
{
	// Define a 3x3 Gaussian filter with which the per-layer blurring will be performed
	const float gaussianFilter[3][3] =
	{
		{ 0.0625f, 0.1250f, 0.0625f },
		{ 0.1250f, 0.2500f, 0.1250f },
		{ 0.0625f, 0.1250f, 0.0625f },
	};

	// The x and y coordinates for which this instance of the kernel is responsible
	short x = threadIdx.x + blockIdx.x * blockDim.x;
	short y = threadIdx.y + blockIdx.y * blockDim.y;
	if (x > imgDims.x || y > imgDims.y)
		return;

	// Initialize the required offsets
	unsigned int pyramidLevelOffset = 0;
	unsigned int offset = x + (y * imgDims.x);
	float redVal, greenVal, blueVal;

	// Throw out every other row and column per iteration
	while (x % 2 == 0 && y % 2 == 0 && imgDims.x > MIN_PYRAMID_SIZE && imgDims.y > MIN_PYRAMID_SIZE)
	{
		redVal = greenVal = blueVal = 0;
		for (int i = offset - imgDims.x, ii = 0; i <= offset + imgDims.x; i += imgDims.x, ++ii)
		{
			for (int j = i - 1, jj = 0; j <= i + 1; ++j, ++jj)
			{
				// If selected pixel is within the bounds, filter it; otherwise, do nothing
				if (j >= pyramidLevelOffset && j < pyramidLevelOffset + (imgDims.x * imgDims.y))
				{
					redVal   += gaussianFilter[ii][jj] * red[j];
					greenVal += gaussianFilter[ii][jj] * green[j];
					blueVal  += gaussianFilter[ii][jj] * blue[j];
				}
			}
		}

		// Update the bounds of the image for the next iteration
		pyramidLevelOffset += imgDims.x * imgDims.y;
		imgDims.x /= 2;
		imgDims.y /= 2;
		x /= 2;
		y /= 2;

		offset = pyramidLevelOffset + x + (y * imgDims.x);

		red[offset]   = (Pixel) redVal;
		green[offset] = (Pixel) greenVal;
		blue[offset]  = (Pixel) blueVal;

		__syncthreads();
	}
}


__global__ void detectEdges(Image red, Image green, Image blue, Image redEdges, Image greenEdges, Image blueEdges,
	short2 imgDims)
{
	// This 3x3 filter is a combination of a horizontal and vertical Sobel filter (to do filtering in one pass)
	const float edgeFilter[3][3] =
	{
		{ -0.333f, -0.333f,  0.000f },
		{ -0.333f,  0.000f,  0.333f },
		{  0.000f,  0.333f,  0.333f },
	};

	// The x and y coordinates for which this instance of the kernel is responsible
	short x = threadIdx.x + (blockIdx.x * blockDim.x);
	short y = threadIdx.y + (blockIdx.y * blockDim.y);
	if (x > imgDims.x || y > imgDims.y)
		return;

	// Initialize the starting offsets
	unsigned int pyramidLevelOffset = 0;
	unsigned int offset = x + (y * imgDims.x);

	float redVal, greenVal, blueVal;
	while (x < imgDims.x && y < imgDims.y && imgDims.x > MIN_PYRAMID_SIZE && imgDims.y > MIN_PYRAMID_SIZE)
	{
		// Calculate the image gradient value at this location for this pyramid level
		redVal = greenVal = blueVal = 0;
		for (int i = offset - imgDims.x, ii = 0; i <= offset + imgDims.x; i += imgDims.x, ++ii)
		{
			for (int j = i - 1, jj = 0; j <= i + 1; ++j, ++jj)
			{
				if (j >= pyramidLevelOffset && j < pyramidLevelOffset + (imgDims.x * imgDims.y))
				{
					redVal   += edgeFilter[ii][jj] * red[j];
					greenVal += edgeFilter[ii][jj] * green[j];
					blueVal  += edgeFilter[ii][jj] * blue[j];
				}
			}
		}

		redVal   = abs(redVal);
		greenVal = abs(greenVal);
		blueVal  = abs(blueVal);

		if (redVal < EDGE_THRESH)
			redVal = 0;
		if (greenVal < EDGE_THRESH)
			greenVal = 0;
		if (blueVal < EDGE_THRESH)
			blueVal = 0;

		// Save the calculated edge intensity
		redEdges[offset]   = (Pixel) redVal;
		greenEdges[offset] = (Pixel) greenVal;
		blueEdges[offset]  = (Pixel) blueVal;

		// Update the offset for the next pyramid level
		pyramidLevelOffset += imgDims.x * imgDims.y;
		imgDims.x /= 2;
		imgDims.y /= 2;
		offset = pyramidLevelOffset + x + (y * imgDims.x);
	}
}


__global__ void alignImages(Image baseEdges, Image alignEdges, short2 imgDims, short2* alignment, short4 threshold, 
	unsigned long long* errSumBuf)
{
	// Adjust the alignment to account for pyramid steps
	if (threadIdx.x == 0 && threadIdx.y == 0 && blockIdx.x == 0 && blockIdx.y == 0)
	{
		alignment->x *= 2;
		alignment->y *= 2;
		memset(errSumBuf, 0, NUM_ALIGN_NEIGHBORS * sizeof(unsigned long long));
	}

	// Determine the position for which this kernel instance is responsible
	const unsigned int x = threadIdx.x + (blockIdx.x * blockDim.x);
	const unsigned int y = threadIdx.y + (blockIdx.y * blockDim.y);
	const unsigned long baseImageOffset = x + (y * imgDims.x);
	const unsigned short blockOffset = threadIdx.x + (threadIdx.y * blockDim.x);

	// Determine whether or not the pixel for which this kernel instance is responsible is in-bounds
	bool inBounds = x > threshold.x && x < threshold.y && y > threshold.z && y < threshold.w;

	// Set up the thread-local error buffer
	__shared__ unsigned int blockErrBuf[THREADS_PER_BLOCK * THREADS_PER_BLOCK];

	__syncthreads();

	// Calculate all of the error values for this pixel location
	for (short i = -1; i <= 1; ++i)
	{
		for (short j = -1; j <= 1; ++j)
		{
			// If the pixel is in-bounds, compute the squared error; otherwise, set the error equal to 0
			if (inBounds)
			{
				unsigned long alignImageOffset = x + alignment->x + i + ((y + alignment->y + j) * imgDims.x);
				blockErrBuf[blockOffset] =
					(unsigned int) powf((short) baseEdges[baseImageOffset] - alignEdges[alignImageOffset], 2);
			}
			else
			{
				blockErrBuf[blockOffset] = 0;
			}

			__syncthreads();

			// Sum up the block-local error into blockErrBuf[0]
			unsigned int blockCutoff = THREADS_PER_BLOCK * THREADS_PER_BLOCK / 2;
			while (blockOffset < blockCutoff)
			{
				blockErrBuf[blockOffset] += blockErrBuf[blockOffset + blockCutoff];
				blockCutoff /= 2;
			}

			// Sum all of the block-local error values into a central buffer
			if (threadIdx.x == 0 && threadIdx.y == 0)
				atomicAdd(errSumBuf + (i + 1) + ((j + 1) * 3), blockErrBuf[0]);

			__syncthreads();
		}
	}

	// Determine which alignment adjustment resulted in the best overlay
	if (threadIdx.x == 0 && threadIdx.y == 0 && blockIdx.x == 0 && blockIdx.y == 0)
	{
		short2 bestAlignment = make_short2(0, 0);
		unsigned long long bestError = errSumBuf[4];
		for (short i = 0; i < NUM_ALIGN_NEIGHBORS; ++i)
		{
			if (errSumBuf[i] < bestError)
			{
				bestAlignment.x = (i / 3) - 1;
				bestAlignment.y = (i % 3) - 1;
				bestError = errSumBuf[i];
			}
		}

		// Apply the calculated adjustment to the alignment
		alignment->x += bestAlignment.x;
		alignment->y += bestAlignment.y;
	}
}


__global__ void produceComposite(Image red, Image green, Image blue, Image composite,
	short2* grOffset, short2* gbOffset, short2 imgDims)
{
	// Compute the x and y values based on the calculated alignment
	short greenX = threadIdx.x + blockIdx.x * blockDim.x;
	short greenY = threadIdx.y + blockIdx.y * blockDim.y;
	short redX = clampShort(greenX + grOffset->x, 0, imgDims.x);
	short redY = clampShort(greenY + grOffset->y, 0, imgDims.y);
	short blueX  = clampShort(greenX + gbOffset->x, 0, imgDims.x);
	short blueY  = clampShort(greenY + gbOffset->y, 0, imgDims.y);
	
	// Use the computed coordinates to calculate the alignments into the image buffers
	unsigned int redOffset = redX + (redY * imgDims.x);
	unsigned int greenOffset = greenX + (greenY * imgDims.x);
	unsigned int blueOffset = blueX + (blueY * imgDims.x);
	unsigned int compOffset = redOffset * NUM_CHANNELS;

	// Combine the three channels into a single image based on the specified color mapping
	composite[compOffset + RED] = (uint8_t) clampFloat(
		red[redOffset]*RED_MAPPING[RED] +
		green[greenOffset]*GREEN_MAPPING[RED] +
		blue[blueOffset]*BLUE_MAPPING[RED], COLOR_MIN, COLOR_MAX);
	composite[compOffset + GREEN] = (uint8_t) clampFloat(
		red[redOffset]*RED_MAPPING[GREEN] +
		green[greenOffset]*GREEN_MAPPING[GREEN] +
		blue[blueOffset]*BLUE_MAPPING[GREEN], COLOR_MIN, COLOR_MAX);
	composite[compOffset + BLUE] = (uint8_t) clampFloat(
		red[redOffset]*RED_MAPPING[BLUE] +
		green[greenOffset]*GREEN_MAPPING[BLUE] +
		blue[blueOffset]*BLUE_MAPPING[BLUE], COLOR_MIN, COLOR_MAX);
}


__global__ void scoreAlignment(Image baseEdges, Image alignEdges, short2 imgDims, short2 alignment, short4 threshold,
	unsigned long long* errSum)
{
	// Zero out the error value
	if (threadIdx.x == 0 && threadIdx.y == 0 && blockIdx.x == 0 && blockIdx.y == 0)
		*errSum = 0;

	// Determine the position for which this kernel instance is responsible
	const unsigned int x = threadIdx.x + (blockIdx.x * blockDim.x);
	const unsigned int y = threadIdx.y + (blockIdx.y * blockDim.y);
	const unsigned long baseImageOffset = x + (y * imgDims.x);
	const unsigned long alignImageOffset = x + alignment.x + ((y + alignment.y) * imgDims.x);
	const unsigned short blockOffset = threadIdx.x + (threadIdx.y * blockDim.x);

	// Determine whether or not the pixel for which this kernel instance is responsible is in-bounds
	bool inBounds = x > threshold.x && x < threshold.y && y > threshold.z && y < threshold.w;

	// Set up the thread-local error buffer
	__shared__ unsigned int blockErrBuf[THREADS_PER_BLOCK * THREADS_PER_BLOCK];

	__syncthreads();

	// If the pixel is in-bounds, compute the squared error; otherwise, set the error equal to 0
	if (inBounds)
	{
		blockErrBuf[blockOffset] = inBounds
			? (unsigned int) powf((short) baseEdges[baseImageOffset] - alignEdges[alignImageOffset], 2)
			: 0;
	}
	else
	{
		blockErrBuf[blockOffset] = 0;
	}

	__syncthreads();

	// Sum up the block-local error into blockErrBuf[0]
	unsigned int blockCutoff = THREADS_PER_BLOCK * THREADS_PER_BLOCK / 2;
	while (blockOffset < blockCutoff)
	{
		blockErrBuf[blockOffset] += blockErrBuf[blockOffset + blockCutoff];
		blockCutoff /= 2;
	}

	// Sum all of the block-local error values into a central buffer
	if (threadIdx.x == 0 && threadIdx.y == 0)
		atomicAdd(errSum, blockErrBuf[0]);

	__syncthreads();
}