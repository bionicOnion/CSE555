/*
 * transitionCost.cpp
 *
 * Author: Robert Miller
 * Last Edit: 4/6/16
 *
 * TODO Explanation
 */


#include <cmath>

#include "consoleProgressBar.hpp"
#include "constants.h"
#include "transitionCost.hpp"


int computeTransitionProbabilities(VideoFrames& video, cv::Mat& costMatrix, float sigmaFactor)
{
  // Allocate space for the cost matrix
  auto frameCount = video.getFrameCount();
  costMatrix = cv::Mat(frameCount, frameCount, CV_64F);

  // Prepare the progress bar (compute the total number of comparisons to be made using Gauss'
  //   method and initialize requisite state)
  ConsoleProgressBar progressBar;
  progressBar.setTitleWidth(PBAR_TITLE_WIDTH);
  progressBar.setTitle("Comparing Frames:");
  auto totalSteps = (frameCount * (frameCount + 1) / 2.0);
  auto completedSteps = 0;

  // Compare each frame to every other frame, computing the SSD error between the pairs and storing
  //   the results into the cost matrix
  for (unsigned i = 0; i < frameCount; ++i)
  {
    for (auto j = i; j < frameCount; ++j)
    {
      costMatrix.at<double>(i, j) = 0;
      for (auto ii = 0; ii < video.getFrame(i).rows; ++ii)
      {
        for (auto jj = 0; jj < video.getFrame(i).cols; ++jj)
        {
          int16_t error =
            video.getFrame(i).at<uint8_t>(ii, jj) - video.getFrame(j).at<uint8_t>(ii, jj);
          costMatrix.at<double>(i, j) += error * error / frameCount; // Divisor keeps values small
        }
      }
      progressBar.printProgress(++completedSteps / totalSteps);
    }
  }
  std::cout << std::endl;

  // Preserve video dynamics
  auto costMatrixMod = cv::Mat(frameCount, frameCount, CV_64F);
  progressBar.setTitle("Preserving Dynamics:");
  totalSteps = (frameCount * (frameCount + 1) / 2.0);
  completedSteps = 0;
  for (unsigned i = 0; i < frameCount; ++i)
  {
    for (auto j = i; j < frameCount; ++j)
    {
      costMatrixMod.at<double>(i, j) = 0;
      for (int k = 0, ii = i - WEIGHT_TAP_COUNT, jj = j - WEIGHT_TAP_COUNT;
           k < (WEIGHT_TAP_COUNT * 2) + 1;
           ++k, ++ii, ++jj)
      {
        if (ii < 0 || ii >= frameCount || jj < 0 || jj >= frameCount)
          costMatrixMod.at<double>(i, j) += WEIGHTS[k];
        else
          costMatrixMod.at<double>(i, j) += WEIGHTS[k] * costMatrix.at<double>(ii, jj);
      }
      costMatrixMod.at<double>(j, i) = costMatrixMod.at<double>(i, j);
      progressBar.printProgress(++completedSteps / totalSteps);
    }
  }
  std::cout << std::endl;

  // Compute average value
  auto avg = 0.0;
  auto avgVals = 0;
  progressBar.setTitle("Compute Average:");
  totalSteps = frameCount * frameCount;
  completedSteps = 0;
  for (unsigned i = 0; i < frameCount; ++i)
  {
    for (auto j = 0; j < frameCount; ++j)
    {
      costMatrix.at<double>(i, j) = costMatrixMod.at<double>(i, j);
      if (costMatrix.at<double>(i, j) != 0)
      {
        avg += costMatrix.at<double>(i, j);
        ++avgVals;
      }
      progressBar.printProgress(++completedSteps / totalSteps);
    }
  }
  avg /= avgVals;
  std::cout << std::endl;

  // Compute probabilities
  progressBar.setTitle("Computing Probabilities:");
  totalSteps = frameCount * frameCount;
  completedSteps = 0;
  for (unsigned i = 0; i < frameCount - 1; ++i)
  {
    for (auto j = 0; j < frameCount; ++j)
    {
      costMatrix.at<double>(i, j) = exp(-costMatrix.at<double>(i + 1, j) / (avg * sigmaFactor));
      progressBar.printProgress(++completedSteps / totalSteps);
    }
  }
  for (auto j = 0; j < frameCount; ++j)
  {
    costMatrix.at<double>(frameCount - 1, j) = 0;
    progressBar.printProgress(++completedSteps / totalSteps);
  }
  std::cout << std::endl;

  // Normalize the probabilities
  progressBar.setTitle("Normalizing Probabilities:");
  totalSteps = frameCount * frameCount * 2;
  completedSteps = 0;
  for (unsigned i = 0; i < frameCount; ++i)
  {
    auto sum = 0.0;
    for (auto j = 0; j < frameCount; ++j)
    {
      sum += costMatrix.at<double>(i, j);
      progressBar.printProgress(++completedSteps / totalSteps);
    }
    for (auto j = 0; j < frameCount; ++j)
    {
      costMatrix.at<double>(i, j) /= sum;
      progressBar.printProgress(++completedSteps / totalSteps);
    }
  }
  std::cout << std::endl;

  return SUCCESS;
}