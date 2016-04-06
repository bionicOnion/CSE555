/*
 * main.cpp
 *
 * Author: Robert Miller
 * Last Edit: 4/6/16
 *
 * The main entry point for the video texture application.
 *
 * TODO Add more of a description here
 */


#include <cstdlib>
#include <ctime>
#include <iomanip>
#include <iostream>
#include <stdint.h>
#include <stdlib.h>

// OpenCV
#include <opencv2/opencv.hpp>

#include "constants.h"
#include "transitionCost.hpp"
#include "video.hpp"


int sampleFromDistribution(cv::Mat distribution, int currIndex);


int main(int argc, char** argv)
{  
  // Validate arguments
  if (argc < MIN_ARG_COUNT)
  {
    std::cout << "Not enough arguments were provided." << std::endl;
    std::cout << "Usage:" <<std::endl;
    std::cout << '\t' << argv[PNAME_INDEX] << " [texture video] [source frame count] [desired frame count]"
      << std::endl;
    return INSUFFICIENT_ARGS;
  }

  // Open the specified video texture
  VideoFrames texVideo;
  auto retCode = texVideo.loadVideo(argv[TEX_VID_INDEX], std::stoi(argv[SRC_FRAME_COUNT_INDEX]));
  if (retCode != SUCCESS)
    return retCode;

  // Compute matrix of transition costs between each frame
  cv::Mat transitionProbs;
  retCode = computeTransitionProbabilities(texVideo, transitionProbs, SIGMA_FACTOR);
  if (retCode != SUCCESS)
    return retCode;

  cv::Mat show;
  cv::resize(transitionProbs, show, cv::Size(400, 400), 0, 0, cv::INTER_NEAREST);
  show.convertTo(show, CV_8UC1, 255);
  cv::imshow("Transitions", show);
  cv::imwrite("transitionProbs.bmp", show);
  cv::waitKey();

  // Generate a new video from the texture
  srand(static_cast<unsigned>(time(nullptr)));
  auto currFrameIdx = 0;
  auto desiredLength = std::stoi(argv[GEN_VID_LEN_INDEX]);
  cv::VideoWriter genVid("TexVideo.avi", texVideo.getFourCC(), texVideo.getFPS(), texVideo.getFrame(0).size());
  for (auto i = 0; i < desiredLength; ++i)
  {
    genVid << texVideo.getFrame(currFrameIdx);
    currFrameIdx = sampleFromDistribution(transitionProbs, currFrameIdx);
  }
  genVid.release();

  return SUCCESS;
}


int sampleFromDistribution(cv::Mat distribution, int currIndex)
{
  auto randVal = static_cast<float>(rand()) / static_cast<float>(RAND_MAX);
  auto selectedIndex = 0;
  auto distSum = distribution.at<double>(currIndex, selectedIndex);
  while (distSum < randVal && selectedIndex < distribution.rows - 1)
  {
    ++selectedIndex;
    distSum += distribution.at<double>(currIndex, selectedIndex);
  }
  return selectedIndex;
}