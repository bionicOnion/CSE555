/*
 * main.cpp
 * Author: Robert Miller
 * Last Edited: 4/27/16
 *
 * TODO
 */


#include <iostream>

#include "argParser.hpp"
#include "imageProcessor.hpp"
#include "imageResource.hpp"
#include "types.hpp"
#include "util.hpp"


int main(int argc, char** argv)
{
    ReturnCode retCode;

    // Parse/validate the provided arguments
    ParamBundle params;
    retCode = parseArguments(&params, argc, argv);
    if (retCode != SUCCESS)
        return PRINT_ERR_MSG(retCode);

//	printParams(&params);

    // Load the specified image/video
    ImageResource input(params.inputType);
    retCode = input.load(params.inputFile);
    if (retCode != SUCCESS)
        return PRINT_ERR_MSG(retCode);

    // Perform processing on the input image/video
    ImageResource output(params.inputType);
    retCode = processImageResource(input, output, params);
    if (retCode != SUCCESS)
        return PRINT_ERR_MSG(retCode);

    // Display and save the output
    output.display();
    retCode = output.save(params.outputFile);
    if (retCode != SUCCESS)
        return PRINT_ERR_MSG(retCode);

    return SUCCESS;
}