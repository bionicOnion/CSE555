/*
 * TODO
 */


#pragma once


#include <cstdint>
#include <string>


// Color/Image data types
typedef uint8_t channel_t;
typedef struct
{
    channel_t r;
    channel_t g;
    channel_t b;
} Color;
typedef Color Pixel;
typedef Pixel* Image;
typedef channel_t* ChannelBuf;


// Parameter types
enum class ColoringMode
{
    AverageColor,
    PixelColors,
    SolidColors,
};

enum class InputType
{
    Image,
    Unset,
    Video,
};

typedef struct
{
    float pointRatio; // Ratio of number of points sampled to resolution of image
    float intensityEdgeWeight; // Weight of pixel intensity vs. edge data for point sampling
    float historicityWeight; // Weight of prior point locations for video processing
    
    ColoringMode mode;
    Color foreground;
    Color background;

    InputType inputType;
    std::string inputFile;
    std::string outputFile;

	bool debug;
	bool timing;
} ParamBundle;


// Return codes
enum ReturnCode : uint16_t
{
	SUCCESS = 0,

	// Argument processing
	ARG_OUT_OF_BOUNDS,
	INSUFFICIENT_ARGS,
	INVALID_ARGUMENT,
	TOO_MANY_INPUT_FILES,
	UNRECOGNIZED_ARG,
	UNRECOGNIZED_COLORING_MODE,
	USAGE_PRINTED,
	NO_INPUT_FILE,

	// File access
	FILE_NOT_OPENED,
	FILE_WRITE_ERROR,
	RESOURCE_UNINITIALIZED,

	// CUDA
	CUDA_ERROR,

	// Development
	NOT_YET_IMPLEMENTED,
    UNRECOGNIZED_INPUT_TYPE,
};