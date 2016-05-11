/*
 * imageResource.hpp
 *
 * Author: Robert Miller
 * Last Edit: 5/10/16
 *
 * ImageResource is a class designed to abstract away the differences between image and video data.
 *   In particular, an ImageResource allows an image to be treated externally as a one-frame video
 *   while internally differentiating between the two content types for the purposes of saving,
 *   loading, and displaying the results.
 */


#pragma once


#include <cstdint>
#include <string>
#include <vector>

// OpenCV
#include <opencv2/opencv.hpp>

#include "types.hpp"


 const short MAX_DIM = 640;


class ImageResource
{
public:
    explicit ImageResource(InputType type);
    explicit ImageResource(InputType type, uint8_t frameRate);

    uint16_t getWidth() const;
    uint16_t getHeight() const;

    uint8_t getFPS(void) const;
    Image getFrame(uint16_t index) const;
    size_t getFrameCount(void) const;

    void addFrame(Image frame, uint16_t width, uint16_t height);

    ReturnCode display(void) const;
    ReturnCode load(std::string filename);
    ReturnCode save(std::string filename);
private:
    InputType type;

    std::vector<cv::Mat> frames;
    uint8_t fps;
};