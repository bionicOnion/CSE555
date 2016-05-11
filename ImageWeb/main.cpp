/*
 * main.cpp
 *
 * Author: Robert Miller
 * Last Edit: 5/10/16
 *
 * The main entry point for the application. Although most of the heavy lifting is done in the
 *   image processor, the main function ultimately coordinates the four major steps in the
 *   lifetime of the application: parsing commands, loading the input file, generating the
 *   rendered image, and saving/displaying that rendered image.
 */


#include "argParser.hpp"
#include "imageProcessor.hpp"
#include "imageResource.hpp"
#include "types.hpp"


int main(int argc, char** argv)
{
    ReturnCode retCode;

    // Parse/validate the provided arguments
    ParamBundle params;
    retCode = parseArguments(&params, argc, argv);
    if (retCode != SUCCESS)
        return retCode;

    if (params.debug)
        printParams(&params);

    // Load the specified image/video
    ImageResource input(params.inputType);
    retCode = input.load(params.inputFile);
    if (retCode != SUCCESS)
        return retCode;

    // Perform processing on the input image/video
    ImageResource output(params.inputType, input.getFPS());
    retCode = processImageResource(input, output, params);
    if (retCode != SUCCESS)
        return retCode;

    // Display and save the output
    output.display();
    retCode = output.save(params.outputFile);
    if (retCode != SUCCESS)
        return retCode;

    return SUCCESS;
}