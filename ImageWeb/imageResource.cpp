/*
 * imageResource.cpp
 *
 * Author: Robert Miller
 * Last Edit: 4/27/16
 *
 * An implementation of the ImageResource class defined in imageResource.hpp
 */


#include "imageResource.hpp"


ImageResource::ImageResource(InputType type) :
    type(type), fps(0)
{}


uint16_t ImageResource::getWidth() const
    { return frames.size() > 0 ? frames[0].cols : 0; }


uint16_t ImageResource::getHeight() const
    { return frames.size() > 0 ? frames[0].rows : 0; }


uint8_t ImageResource::getFPS() const
    { return fps; }


// TODO convert from BGR to RGB color space
Image ImageResource::getFrame(uint16_t index) const
    { return index < frames.size() ? Image(frames[index].ptr()) : nullptr; }


size_t ImageResource::getFrameCount(void) const
    { return frames.size(); }


void ImageResource::addFrame(Image frame, uint16_t width, uint16_t height)
{
	cv::Mat mat;
	mat.create(height, width, CV_8UC3);
	memcpy(mat.ptr(), frame, width * height * sizeof(Pixel));
	frames.push_back(mat);
}


ReturnCode ImageResource::display(void) const
{
	if (frames.size() < 1)
		return RESOURCE_UNINITIALIZED;

	if (type == InputType::Image)
	{
		cv::imshow("Image", frames[0]);
		cv::waitKey();
	}
	else if (type == InputType::Video)
	{
		return NOT_YET_IMPLEMENTED;
	}
	else
		return UNRECOGNIZED_INPUT_TYPE;

    return SUCCESS;
}


ReturnCode ImageResource::load(std::string filename)
{
    // Empty out any other loaded image resource
    frames.clear();

    // Load either an image or a video as specified in the constructor
    if (type == InputType::Image)
    {
        frames.push_back(cv::imread(filename));
        if (!frames[0].data)
            return FILE_NOT_OPENED;
    }
    else if (type == InputType::Video)
    {
        // Open the specified file
        cv::VideoCapture textureImageResource(filename);
        if (!textureImageResource.isOpened())
            return FILE_NOT_OPENED;

        // Retrieve relevant data from input file
        fps = static_cast<uint8_t>(textureImageResource.get(CV_CAP_PROP_FPS));
		frames.resize(static_cast<size_t>(textureImageResource.get(CV_CAP_PROP_FRAME_COUNT)));

        // Load the ImageResource into a vector of frames
        auto framesRead = -1;
        while (textureImageResource.read(frames[framesRead]))
            ++framesRead;

        // Release the file now that it is no longer needed
        textureImageResource.release();
    }
    else
        return UNRECOGNIZED_INPUT_TYPE;

    return SUCCESS;
}


ReturnCode ImageResource::save(std::string filename)
{
    if (type == InputType::Image)
    {
        if (!cv::imwrite(filename, frames[0]))
            return FILE_WRITE_ERROR;
    }
    else if (type == InputType::Video)
    {
        return NOT_YET_IMPLEMENTED;
    }
    else
        return UNRECOGNIZED_INPUT_TYPE;

    return SUCCESS;
}