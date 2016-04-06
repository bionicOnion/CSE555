/*
 * video.cpp
 *
 * Author: Robert Miller
 * Last Edit: 1/6/16
 *
 * An implementation of the Video class defined in video.hpp
 */


#include "constants.h"
#include "video.hpp"


VideoFrames::VideoFrames() :
  fps(0)
{}


int VideoFrames::loadVideo(std::string videoFile)
{
  // Open the spcified video file
  cv::VideoCapture textureVideo(videoFile);
  if (!textureVideo.isOpened())
  {
    std::cout << "Failed to open the specified file " << videoFile << '.' << std::endl;
    return FILE_NOT_OPENED;
  }

  // Determine the frame rate & fourcc (for later playback)
  fps = static_cast<uint8_t>(textureVideo.get(CV_CAP_PROP_FPS));
  fourcc = static_cast<int>(textureVideo.get(CV_CAP_PROP_FOURCC));

  // Load the video into a vector of frames
  auto framesRead = -1;
  auto frameCount = MIN(static_cast<unsigned int>(textureVideo.get(CV_CAP_PROP_FRAME_COUNT)),
    MAX_FRAMES);
  frames.resize(frameCount);
  cv::Mat frame;
  while (textureVideo.read(frame) && ++framesRead < frameCount)
    cv::resize(frame, frames[framesRead], cv::Size(), VID_SCALE_FACTOR, VID_SCALE_FACTOR);

  // Release the video file now that it is no longer needed
  textureVideo.release();

  return SUCCESS;
}



cv::Mat VideoFrames::getFrame(unsigned int index)
  { return index < frames.size() ? frames[index] : cv::Mat(); }



size_t VideoFrames::getFrameCount(void) const
  { return frames.size(); }


uint8_t VideoFrames::getFPS() const
  { return fps; }



int VideoFrames::getFourCC(void) const
  { return fourcc; }