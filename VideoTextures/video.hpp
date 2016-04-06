/*
 * video.hpp
 *
 * Author: Robert Miller
 * Last Edit: 4/6/16
 *
 * TODO Fill in a description of the class
 */


#pragma once


 #include <string>
 #include <vector>

// OpenCV
#include <opencv2/opencv.hpp>


class VideoFrames
{
public:
  VideoFrames();

  uint8_t getFPS(void) const;
  int getFourCC(void) const;
  cv::Mat getFrame(unsigned int);
  size_t getFrameCount(void) const;
  int loadVideo(std::string);
private:
  int fourcc;
  std::vector<cv::Mat> frames;
  uint8_t fps;
};