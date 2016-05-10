/*
 * TODO
 */


#include <ctime>
#include <iostream>
#include <sstream>

#include "argParser.hpp"
#include "util.hpp"


// Forward declarations
ReturnCode parse(ParamBundle* params, int argc, char** argv);
void printUsage(std::string programName);
ReturnCode validate(ParamBundle* params);


ReturnCode parseArguments(ParamBundle* params, int argc, char** argv)
{
    ReturnCode retCode;

    // Fill in default values
    params->pointRatio = DEFAULT_POINT_RATIO;
    params->intensityEdgeWeight = DEFAULT_INTENSITY_EDGE_WEIGHT;
    params->historicityWeight = DEFAULT_HISTORICITY_WEIGHT;
    params->mode = DEFAULT_COLORING_MODE;
    params->foreground = DEFAULT_FOREGROUND;
    params->background = DEFAULT_BACKGROUND;
	params->inputType = InputType::Unset;
	params->debug = false;
	params->timing = false;
	params->seed = time(nullptr);

    // Insert provided values into the parameter bundle
    retCode = parse(params, argc, argv);
    if (retCode != SUCCESS)
        return retCode;

    // Validate the contents of the parameter bundle
    retCode = validate(params);
    if (retCode != SUCCESS)
        return retCode;

    return SUCCESS;
}


ReturnCode parse(ParamBundle* params, int argc, char** argv)
{
    for (auto i = 1; i < argc; ++i)
    {
        if (BACKGROUND_COLOR_ARGS.find(argv[i]) != BACKGROUND_COLOR_ARGS.end())
        {
			if (++i >= argc)
				return PRINT_ERR_MSG(INSUFFICIENT_ARGS);
			params->background.r = std::stoi(argv[i]) / 255.0;
			if (++i >= argc)
				return PRINT_ERR_MSG(INSUFFICIENT_ARGS);
			params->background.g = std::stoi(argv[i]) / 255.0;
			if (++i >= argc)
				return PRINT_ERR_MSG(INSUFFICIENT_ARGS);
			params->background.b = std::stoi(argv[i]) / 255.0;
        }
        else if (COLORING_MODE_ARGS.find(argv[i]) != COLORING_MODE_ARGS.end())
        {
			if (++i >= argc)
				return PRINT_ERR_MSG(INSUFFICIENT_ARGS);

			if (BLENDED_COLORING_MODE.find(argv[i]) != BLENDED_COLORING_MODE.end())
				params->mode = ColoringMode::BlendedColor;
			else if (CENTROID_COLORING_MODE.find(argv[i]) != CENTROID_COLORING_MODE.end())
				params->mode = ColoringMode::CentroidColor;
            else if (PIXEL_COLORING_MODE.find(argv[i]) != PIXEL_COLORING_MODE.end())
                params->mode = ColoringMode::PixelColors;
            else if (SOLID_COLORING_MODE.find(argv[i]) != SOLID_COLORING_MODE.end())
                params->mode = ColoringMode::SolidColors;
			else
				return PRINT_ERR_MSG(UNRECOGNIZED_COLORING_MODE);
        }
		else if (DEBUG_MODE_ARGS.find(argv[i]) != DEBUG_MODE_ARGS.end())
		{
			params->debug = true;
		}
        else if (FOREGROUND_COLOR_ARGS.find(argv[i]) != FOREGROUND_COLOR_ARGS.end())
        {
			// TODO Handle possible exceptions
			if (++i >= argc)
				return PRINT_ERR_MSG(INSUFFICIENT_ARGS);
			params->foreground.r = std::stoi(argv[i]) / 255.0;
			if (++i >= argc)
				return PRINT_ERR_MSG(INSUFFICIENT_ARGS);
			params->foreground.g = std::stoi(argv[i]) / 255.0;
			if (++i >= argc)
				return PRINT_ERR_MSG(INSUFFICIENT_ARGS);
			params->foreground.b = std::stoi(argv[i]) / 255.0;
        }
        else if (HELP_ARGS.find(argv[i]) != HELP_ARGS.end())
        {
			printUsage(argv[0]);
			return PRINT_ERR_MSG(USAGE_PRINTED);
        }
        else if (HISTORICITY_ARGS.find(argv[i]) != HISTORICITY_ARGS.end())
        {
			if (++i >= argc)
				return PRINT_ERR_MSG(INSUFFICIENT_ARGS);

			try
			{
				params->historicityWeight = std::stof(argv[i]);
			}
			catch (std::invalid_argument)
			{
				return PRINT_ERR_MSG(INVALID_ARGUMENT);
			}
			catch (std::out_of_range)
			{
				params->historicityWeight = 1;
			}
        }
        else if (IMAGE_FILE_ARGS.find(argv[i]) != IMAGE_FILE_ARGS.end())
        {
			if (params->inputType != InputType::Unset)
				return PRINT_ERR_MSG(TOO_MANY_INPUT_FILES);
			if (++i >= argc)
				return PRINT_ERR_MSG(INSUFFICIENT_ARGS);

            params->inputType = InputType::Image;
            params->inputFile = argv[i];
        }
        else if (INTENSITY_EDGE_WEIGHT_ARGS.find(argv[i]) != INTENSITY_EDGE_WEIGHT_ARGS.end())
        {
			if (++i >= argc)
				return PRINT_ERR_MSG(INSUFFICIENT_ARGS);

			try
			{
				params->intensityEdgeWeight = std::stof(argv[i]);
			}
			catch (std::invalid_argument)
			{
				return PRINT_ERR_MSG(INVALID_ARGUMENT);
			}
			catch (std::out_of_range)
			{
				params->intensityEdgeWeight = 1;
			}
        }
		else if (OUTPUT_FILE_ARGS.find(argv[i]) != OUTPUT_FILE_ARGS.end())
		{
			if (++i >= argc)
				return PRINT_ERR_MSG(INSUFFICIENT_ARGS);

			params->outputFile = argv[i];
		}
        else if (POINT_RATIO_ARGS.find(argv[i]) != POINT_RATIO_ARGS.end())
        {
			if (++i >= argc)
				return PRINT_ERR_MSG(INSUFFICIENT_ARGS);

			try
			{
				params->pointRatio = std::stof(argv[i]);
			}
			catch (std::invalid_argument)
			{
				return PRINT_ERR_MSG(INVALID_ARGUMENT);
			}
			catch (std::out_of_range)
			{
				params->pointRatio = 1;
			}
		}
		else if (SEED_ARGS.find(argv[i]) != SEED_ARGS.end())
		{
			if (++i >= argc)
				return PRINT_ERR_MSG(INSUFFICIENT_ARGS);

			try
			{
				params->seed = std::stol(argv[i]);
			}
			catch (std::invalid_argument)
			{
				return PRINT_ERR_MSG(INSUFFICIENT_ARGS);
			}
			catch (std::out_of_range)
			{
				params->seed = 0;
			}
		}
		else if (TIMING_ARGS.find(argv[i]) != TIMING_ARGS.end())
		{
			params->timing = true;
		}
        else if (VIDEO_FILE_ARGS.find(argv[i]) != VIDEO_FILE_ARGS.end())
        {
			if (params->inputType != InputType::Unset)
				return PRINT_ERR_MSG(TOO_MANY_INPUT_FILES);
			if (++i >= argc)
				return PRINT_ERR_MSG(INSUFFICIENT_ARGS);

            params->inputType = InputType::Video;
            params->inputFile = argv[i];
        }
        else
        {
			std::cout << "The argument " << argv[i] << " was not recognized" << std::endl;
			return PRINT_ERR_MSG(UNRECOGNIZED_ARG);
        }
    }

    return SUCCESS;
}


ReturnCode validate(ParamBundle* params)
{
	if (params->pointRatio < 0 || params->pointRatio > 1)
		return PRINT_ERR_MSG(ARG_OUT_OF_BOUNDS);
	if (params->intensityEdgeWeight < 0 || params->intensityEdgeWeight > 1)
		return PRINT_ERR_MSG(ARG_OUT_OF_BOUNDS);
	if (params->historicityWeight < 0 || params->historicityWeight > 1)
		return PRINT_ERR_MSG(ARG_OUT_OF_BOUNDS);
	if (params->inputType == InputType::Unset || params->inputFile == "")
		return PRINT_ERR_MSG(NO_INPUT_FILE);

	if (params->outputFile == "")
    {
        // Build a default name for the output file
		// TODO clean this up a bit
		std::istringstream iss(params->inputFile);
		std::vector<std::string> inputFileNameVec;
		std::string token;
		while (std::getline(iss, token, '.'))
			inputFileNameVec.push_back(token);
		if (inputFileNameVec.size() < 2)
			return PRINT_ERR_MSG(INVALID_ARGUMENT);
        auto fileExtension = inputFileNameVec[inputFileNameVec.size() - 1];
        auto fileName = inputFileNameVec[inputFileNameVec.size() - 2];
		auto strCutIndex = 0;
		while (fileName[strCutIndex] == '\\' || fileName[strCutIndex] == '/')
			++strCutIndex;
		fileName = fileName.substr(strCutIndex);
		params->outputFile = fileName + "_web." + fileExtension;
    }

	return SUCCESS;
}


void printUsage(std::string programName)
{
    std::cout << "Usage:" << std::endl;
    std::cout << "  " << programName << " [-i/--image value]/[-v/--video value]" << std::endl;
    std::cout << "Optional Parameters:" << std::endl;
    std::cout << "  Background Color:                [-b/--background r g b]" << std::endl;
    std::cout << "  Coloring Mode:                   [-c/--coloringMode]" << std::endl;
	std::cout << "    [b/blend]: Similar to Pixel Mode, but with filled triangles" << std::endl;
	std::cout << "    [c/centroid]: Filled triangles with color sampled from centroid" << std::endl;
    std::cout << "    [p/pix/pixel]: Foreground color is sampled from input image" << std::endl;
    std::cout << "    [s/solid]: Foreground/background colors are user-specified" << std::endl;
	std::cout << "  Debug Mode:                      [-d/--debug]" <<std::endl;
	std::cout << "  Foreground Color:                [-f/--foreground r g b]" << std::endl;
	std::cout << "  Output File:                     [-o/--outputFile filename]" << std::endl;
    std::cout << "  Point Location Historicity:      [-h/--historicity value]" << std::endl;
    std::cout << "  Intensity vs. Edges Weight:      [-w/--weightRatio value]" << std::endl;
	std::cout << "  Point Count to Resolution Ratio: [-r/--pointRatio value]" << std::endl;
	std::cout << "  RNG Seed:                        [-s/--seed value]" << std::endl;
	std::cout << "  Record Timings:                  [-t/--time]" << std::endl;
}


void printParams(ParamBundle* params)
{
	std::cout << "Parameters:" << std::endl;
	std::cout << "  Point ratio:    " << params->pointRatio << std::endl;
	std::cout << "  Intensity/Edge: " << params->intensityEdgeWeight << std::endl;
	std::cout << "  Historicity:    " << params->historicityWeight << std::endl;
	std::string mode;
	if (params->mode == ColoringMode::BlendedColor)
		mode = "Blended Colors";
	else if (params->mode == ColoringMode::CentroidColor)
		mode = "Centroid Color";
	else if (params->mode == ColoringMode::PixelColors)
		mode = "Pixel Color";
	else if (params->mode == ColoringMode::SolidColors)
		mode = "Solid Color";
	else
		mode = "Unrecognized Coloring Mode";
	std::cout << "  Coloring Mode:  " << mode << std::endl;
	std::cout << "  Foreground:     ("
		<< static_cast<unsigned>(params->foreground.r) << ", "
		<< static_cast<unsigned>(params->foreground.g) << ", "
		<< static_cast<unsigned>(params->foreground.b) << ")" << std::endl;
	std::cout << "  Background:     ("
		<< static_cast<unsigned>(params->background.r) << ", "
		<< static_cast<unsigned>(params->background.g) << ", "
		<< static_cast<unsigned>(params->background.b) << ")" << std::endl;
	std::string inType;
	if (params->inputType == InputType::Image)
		inType = "Image";
	else if (params->inputType == InputType::Video)
		inType = "Video";
	else
		inType = "Unset";
	std::cout << "  Input type:     " << inType << std::endl;
	std::cout << "  Input file:     " << params->inputFile << std::endl;
	std::cout << "  Output file:    " << params->outputFile << std::endl;
	std::cout << std::endl;
}