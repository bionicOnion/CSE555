/*
 * argParser.hpp
 *
 * Author: Robert Miller
 * Last Edit: 5/10/16
 *
 * The parseArguments function is resposible for reading the arguments provided on the command line
 *   and putting the relevant information into a ParamBundle for ease of use elsewhere in the
 *   application. This file contains the signature for that function, a function for displaying the
 *   parameters after parsing has been completed, and a set of definitions for both argument flags
 *   and default values (where applicable).
 */


#pragma once


#include <unordered_set>
#include <string>

#include "types.hpp"


// Default parameter values
const float DEFAULT_POINT_RATIO = 0.005f;
const float DEFAULT_INTENSITY_EDGE_WEIGHT = 0.005f;
const float DEFAULT_HISTORICITY_WEIGHT = 0.1f;
const ColoringMode DEFAULT_COLORING_MODE = ColoringMode::SolidColors;
const Color DEFAULT_FOREGROUND = Color { 1, 1, 1 }; // Default to white
const Color DEFAULT_BACKGROUND = Color { 0, 0, 0 }; // Default to black

// Argument keys
const std::unordered_set<std::string> BACKGROUND_COLOR_ARGS { "-b", "--background" };
const std::unordered_set<std::string> COLORING_MODE_ARGS{ "-c", "--coloringMode" };
const std::unordered_set<std::string> DEBUG_MODE_ARGS{ "-d", "--debug" };
const std::unordered_set<std::string> FOREGROUND_COLOR_ARGS { "-f", "--foreground" };
const std::unordered_set<std::string> HELP_ARGS { "--help" };
const std::unordered_set<std::string> HISTORICITY_ARGS { "-h", "--historicity" };
const std::unordered_set<std::string> IMAGE_FILE_ARGS { "-i", "--image" };
const std::unordered_set<std::string> INTENSITY_EDGE_WEIGHT_ARGS{ "-w", "--weightRatio" };
const std::unordered_set<std::string> OUTPUT_FILE_ARGS{ "-o", "--outputFile" };
const std::unordered_set<std::string> POINT_RATIO_ARGS{ "-r", "--pointRatio" };
const std::unordered_set<std::string> SEED_ARGS { "-s", "--seed" };
const std::unordered_set<std::string> TIMING_ARGS{ "-t", "--time" };
const std::unordered_set<std::string> VIDEO_FILE_ARGS { "-v", "--video" };

// Coloring mode options
const std::unordered_set<std::string> CENTROID_COLORING_MODE{ "c", "centroid" };
const std::unordered_set<std::string> PIXEL_COLORING_MODE { "p", "pix", "pixel" };
const std::unordered_set<std::string> SOLID_COLORING_MODE { "s", "solid" };


ReturnCode parseArguments(ParamBundle* params, int argc, char** argv);
void printParams(ParamBundle* params);