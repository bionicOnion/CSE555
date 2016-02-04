/*
 * main.cu
 * Author: Robert Miller
 * Last Edited: 2/2/16
 *
 * The main entry point of the application which will align and composite images representing the red, green, and blue
 *   color channels of a single image by employing image pyramids.
 *
 * Images are loaded and displayed using the utilities provided by OpenCV, but most of the processing work has been
 *   handed off to the GPU through the use of CUDA C. This has the marked disadvantage that this code can only run on
 *   systems with CUDA-compliant hardware (namely GPUs sold by Nvidia), but the framework has been used here as a
 *   learning tool. Future projects may be implemented using OpenCL to help ensure greater compatibility.
 */


#include <chrono>
#include <iostream>
#include <stdint.h>

// CUDA
#include <cuda_runtime.h>

// OpenCV
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include "accelerator.cuh"
#include "constants.cuh"


// A macro for clean error handling without unduly cluttering the code
#define CUDA_CALL(CALL, ERR_CODE, LINE)																					\
{																														\
	cudaError_t retCode = (CALL);																						\
	if (retCode != cudaSuccess)																							\
	{																													\
		std::cout << "A CUDA call has failed on line " << (LINE) <<  ": " << cudaGetErrorString(retCode) << std::endl;	\
		return (ERR_CODE);																								\
	}																													\
}


struct PyramidLevel
{
	unsigned int offset;
	short2 dims;
};


int DEBUG_displayImage(uint8_t* devPtr, short2 imgDims, char* title)
{
	auto buf = malloc(imgDims.x * imgDims.y);
	CUDA_CALL(cudaMemcpy(buf, devPtr, imgDims.x * imgDims.y, cudaMemcpyDeviceToHost), HOST_CPY_FAIL, __LINE__);
	cv::Mat img(imgDims.y, imgDims.x, CV_8UC1, buf);
	auto smallSize = imgDims.x > imgDims.y
		? cv::Size(MAX_SMALL_IMG_DIM, (short) (imgDims.y * (((float) MAX_SMALL_IMG_DIM) / imgDims.x)))
		: cv::Size((short) (imgDims.x * (((float) MAX_SMALL_IMG_DIM) / imgDims.y)), MAX_SMALL_IMG_DIM);
	if (imgDims.x < MAX_SMALL_IMG_DIM && imgDims.y < MAX_SMALL_IMG_DIM)
		smallSize = cv::Size(imgDims.x, imgDims.y);
	cv::resize(img, img, smallSize);
	cv::imshow(title, img);
	cv::waitKey();
	return SUCCESS;
}


int main(int argc, char** argv)
{
	// If too few arguments were provided, print a usage message and exit
	if (argc < EXPECTED_NUM_ARGS)
	{
		std::cout << "Too few arguments provided." << std::endl << "\tUsage: " << argv[PNAME_ARG_INDEX] << " [image]";
		return INCORRECT_USAGE;
	}

	// Record the time before starting computation
	auto startTime = std::chrono::high_resolution_clock::now();

	// Load the specified source image and calculate the dimensions for the resulting composite image
	auto sourceImage = cv::imread(argv[IMG_ARG_INDEX], CV_LOAD_IMAGE_GRAYSCALE);
	auto compImgDims = make_short2(sourceImage.cols, sourceImage.rows / NUM_CHANNELS);
	dim3 blockSize(compImgDims.x / THREADS_PER_BLOCK, compImgDims.y / THREADS_PER_BLOCK);
	dim3 threadSize(THREADS_PER_BLOCK, THREADS_PER_BLOCK);

	// Separate the image into the three component channels
	if (!sourceImage.isContinuous())
	{
		std::cout << "This implementation does not support discontinuous images." << std::endl;
		return DISCONT_MATRIX;
	}
	auto channelSize = compImgDims.x * compImgDims.y;
	auto blueChannel = sourceImage.ptr();
	auto greenChannel = blueChannel + channelSize;
	auto redChannel = greenChannel + channelSize;

	// Allocate space on the GPU to copy the images and other required variables to
	// For the three channel buffers, enough space is allocated to construct a full image pyramid
	// For the composite buffer, enough space is allocated to construct a 3-channel image
	Image dev_red, dev_green, dev_blue, dev_comp;
	Image dev_redEdges, dev_greenEdges, dev_blueEdges;
	short2 *dev_alignGR, *dev_alignGB;
	unsigned long long* dev_errorSum;
	CUDA_CALL(cudaMalloc(&dev_red, channelSize * 4 / 3), DEV_ALLOC_FAIL, __LINE__);
	CUDA_CALL(cudaMalloc(&dev_green, channelSize * 4 / 3), DEV_ALLOC_FAIL, __LINE__);
	CUDA_CALL(cudaMalloc(&dev_blue, channelSize * 4 / 3), DEV_ALLOC_FAIL, __LINE__);
	CUDA_CALL(cudaMalloc(&dev_redEdges, channelSize * 4 / 3), DEV_ALLOC_FAIL, __LINE__);
	CUDA_CALL(cudaMalloc(&dev_greenEdges, channelSize * 4 / 3), DEV_ALLOC_FAIL, __LINE__);
	CUDA_CALL(cudaMalloc(&dev_blueEdges, channelSize * 4 / 3), DEV_ALLOC_FAIL, __LINE__);
	CUDA_CALL(cudaMalloc(&dev_comp, channelSize * NUM_CHANNELS), DEV_ALLOC_FAIL, __LINE__);
	CUDA_CALL(cudaMalloc(&dev_alignGR, sizeof(short2)), DEV_ALLOC_FAIL, __LINE__);
	CUDA_CALL(cudaMalloc(&dev_alignGB, sizeof(short2)), DEV_ALLOC_FAIL, __LINE__);
	CUDA_CALL(cudaMalloc(&dev_errorSum, sizeof(unsigned long long) * NUM_ALIGN_NEIGHBORS), DEV_ALLOC_FAIL, __LINE__);

	// Copy data from the provided image into the GPU buffers
	CUDA_CALL(cudaMemcpy(dev_red, redChannel, channelSize, cudaMemcpyHostToDevice), DEV_CPY_FAIL, __LINE__);
	CUDA_CALL(cudaMemcpy(dev_green, greenChannel, channelSize, cudaMemcpyHostToDevice), DEV_CPY_FAIL, __LINE__);
	CUDA_CALL(cudaMemcpy(dev_blue, blueChannel, channelSize, cudaMemcpyHostToDevice), DEV_CPY_FAIL, __LINE__);
	
	// Prepare CUDA timing variables
	cudaEvent_t start, edges, pyramids, alignment, finish;
	CUDA_CALL(cudaEventCreate(&start), GPU_TIMING_FAIL, __LINE__);
	CUDA_CALL(cudaEventCreate(&edges), GPU_TIMING_FAIL, __LINE__);
	CUDA_CALL(cudaEventCreate(&pyramids), GPU_TIMING_FAIL, __LINE__);
	CUDA_CALL(cudaEventCreate(&alignment), GPU_TIMING_FAIL, __LINE__);
	CUDA_CALL(cudaEventCreate(&finish), GPU_TIMING_FAIL, __LINE__);

	// Detect edges and generate image pyramids
	CUDA_CALL(cudaEventRecord(start, 0), GPU_TIMING_FAIL, __LINE__);
	generateImagePyramids<<<blockSize, threadSize>>>(dev_red, dev_green, dev_blue, compImgDims);
	CUDA_CALL(cudaEventRecord(pyramids, 0), GPU_TIMING_FAIL, __LINE__);
	detectEdges<<<blockSize, threadSize>>>(dev_red, dev_green, dev_blue, dev_redEdges, dev_greenEdges, dev_blueEdges,
		compImgDims);
	CUDA_CALL(cudaEventRecord(edges, 0), GPU_TIMING_FAIL, __LINE__);

	// Calculate the sizes and offsets of each of the image pyramid levels
	PyramidLevel levels[NUM_ALIGN_LEVELS];
	levels[0].offset = 0;
	levels[0].dims = make_short2(compImgDims.x, compImgDims.y);
	for (short i = 1; i < NUM_ALIGN_LEVELS; ++i)
	{
		levels[i].offset = levels[i - 1].offset + (levels[i - 1].dims.x * levels[i - 1].dims.y);
		levels[i].dims = make_short2(levels[i - 1].dims.x / 2, levels[i - 1].dims.y / 2);
	}

	// Compute the alignments for the color channels
	short2 startAlign = make_short2(0, 0);
	CUDA_CALL(cudaMemcpy(dev_alignGR, &startAlign, sizeof(short2), cudaMemcpyHostToDevice), DEV_CPY_FAIL, __LINE__);
	CUDA_CALL(cudaMemcpy(dev_alignGB, &startAlign, sizeof(short2), cudaMemcpyHostToDevice), DEV_CPY_FAIL, __LINE__);
	for (short i = NUM_ALIGN_LEVELS - 1; i >= 0; --i)
	{
		if (levels[i].dims.x < MIN_PYRAMID_SIZE || levels[i].dims.y < MIN_PYRAMID_SIZE)
			continue;
		
		// Perform image alignment
		alignImages<<<blockSize, threadSize>>>(dev_greenEdges + levels[i].offset,
			dev_redEdges + levels[i].offset, levels[i].dims, dev_alignGR, dev_errorSum);
		alignImages<<<blockSize, threadSize>>>(dev_greenEdges + levels[i].offset,
			dev_blueEdges + levels[i].offset, levels[i].dims, dev_alignGB, dev_errorSum);
	}

	// Finish producing the composite and perform post-processing
	CUDA_CALL(cudaEventRecord(alignment, 0), GPU_TIMING_FAIL, __LINE__);
	produceComposite<<<blockSize, threadSize>>>(dev_red, dev_green, dev_blue, dev_comp, dev_alignGR, dev_alignGB,
		compImgDims);
	CUDA_CALL(cudaEventRecord(finish, 0), GPU_TIMING_FAIL, __LINE__);
	CUDA_CALL(cudaEventSynchronize(finish), GPU_TIMING_FAIL, __LINE__);

	// Record the elapsed timing
	float pyramidTime, edgeTime, alignmentTime, compositingTime, totalTime;
	CUDA_CALL(cudaEventElapsedTime(&pyramidTime, start, pyramids), GPU_TIMING_FAIL, __LINE__);
	CUDA_CALL(cudaEventElapsedTime(&edgeTime, pyramids, edges), GPU_TIMING_FAIL, __LINE__);
	CUDA_CALL(cudaEventElapsedTime(&alignmentTime, edges, alignment), GPU_TIMING_FAIL, __LINE__);
	CUDA_CALL(cudaEventElapsedTime(&compositingTime, alignment, finish), GPU_TIMING_FAIL, __LINE__);
	CUDA_CALL(cudaEventElapsedTime(&totalTime, start, finish), GPU_TIMING_FAIL, __LINE__);

	// Copy the composite back from the device
	auto compBuf = malloc(channelSize * NUM_CHANNELS);
	CUDA_CALL(cudaMemcpy(compBuf, dev_comp, channelSize * NUM_CHANNELS, cudaMemcpyDeviceToHost), HOST_CPY_FAIL, __LINE__);
	cv::Mat compositeImage(compImgDims.y, compImgDims.x, CV_8UC3, compBuf);

	// Print timings and alignments
	short2 grAlign, gbAlign;
	CUDA_CALL(cudaMemcpy(&grAlign, dev_alignGR, sizeof(short2), cudaMemcpyDeviceToHost), HOST_CPY_FAIL, __LINE__);
	CUDA_CALL(cudaMemcpy(&gbAlign, dev_alignGB, sizeof(short2), cudaMemcpyDeviceToHost), HOST_CPY_FAIL, __LINE__);
	auto endTime = std::chrono::high_resolution_clock::now();
	auto elapsedTime = std::chrono::duration_cast<std::chrono::duration<float>>(endTime - startTime);
	std::cout << "Alignment computed:" << std::endl;
	std::cout << "    Green/Red:  (" << grAlign.x << ", " << grAlign.y << ")" << std::endl;
	std::cout << "    Green/Blue: (" << gbAlign.x << ", " << gbAlign.y << ")" << std::endl;
	std::cout << "Composition took " << elapsedTime.count() << " seconds." << std::endl;
	std::cout << "    Pyramid creation time: " << pyramidTime << " ms" << std::endl;
	std::cout << "    Edge detection time: " << edgeTime << " ms" << std::endl;
	std::cout << "    Alignment time: " << alignmentTime << " ms" << std::endl;
	std::cout << "    Compositing time: " << compositingTime << " ms" << std::endl;
	std::cout << std::endl;
	std::cout << "  Total GPU time: " << totalTime << " ms" << std::endl;

	// Free CUDA timing variables
	CUDA_CALL(cudaEventDestroy(start), GPU_TIMING_FAIL, __LINE__);
	CUDA_CALL(cudaEventDestroy(edges), GPU_TIMING_FAIL, __LINE__);
	CUDA_CALL(cudaEventDestroy(pyramids), GPU_TIMING_FAIL, __LINE__);
	CUDA_CALL(cudaEventDestroy(alignment), GPU_TIMING_FAIL, __LINE__);
	CUDA_CALL(cudaEventDestroy(finish), GPU_TIMING_FAIL, __LINE__);

	// Free all allocated device memory
	cudaFree(dev_red);
	cudaFree(dev_green);
	cudaFree(dev_blue);
	cudaFree(dev_redEdges);
	cudaFree(dev_greenEdges);
	cudaFree(dev_blueEdges);
	cudaFree(dev_comp);
	cudaFree(dev_alignGR);
	cudaFree(dev_alignGB);
	cudaFree(dev_errorSum);

	// Generate a reduced-size version of the image to display more easily
	cv::Mat compSmall;
	auto smallSize = compImgDims.x > compImgDims.y
		? cv::Size(MAX_SMALL_IMG_DIM, (short)(compImgDims.y * (((float) MAX_SMALL_IMG_DIM) / compImgDims.x)))
		: cv::Size((short)(compImgDims.x * (((float) MAX_SMALL_IMG_DIM) / compImgDims.y)), MAX_SMALL_IMG_DIM);
	cv::resize(compositeImage, compSmall, smallSize);

	// Save a full-size and small version of the composite image
	cv::imwrite(std::string(argv[IMG_ARG_INDEX]) + ".bmp", compositeImage);
	cv::imwrite(std::string(argv[IMG_ARG_INDEX]) + "_small.bmp", compSmall);

	// Display the resulting image
	cv::imshow(argv[IMG_ARG_INDEX], compSmall);
	cv::waitKey();

	// Free remaining memory
	free(compBuf);

	return SUCCESS;
}