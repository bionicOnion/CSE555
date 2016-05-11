/*
 * consoleProgressBar.hpp
 *
 * Author: Robert Miller
 * Last Edit: 3/30/16
 *
 * A small, cross-platform utility for printing a progress bar to the command line.
 * 
 * This code has been written on Windows and has yet to be tested on either OS X or Linux; the
 *   implementations for those platforms are provisional and may be buggy or incomplete.
 */


#pragma once


#include <cstdint>
#include <string>


const uint8_t DEFAULT_PBAR_WIDTH = 64;
const uint8_t PERCENT_WIDTH = 10;


class ConsoleProgressBar
{
public:
  ConsoleProgressBar();
  explicit ConsoleProgressBar(std::string);

  void print(void);
  void printProgress(double);
  void setProgress(double);
  void setTitle(std::string);
  void setTitleWidth(uint8_t);
private:
  double progress;
  std::string title;
  uint8_t progressBarWidth;
  uint8_t titleWidth;

  void calculateProgressBarWidth(void);
};