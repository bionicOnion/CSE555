/*
 * consoleProgressBar.cpp
 *
 * Author: Robert Miller
 * Last Edit: 5/10/16
 *
 * An implementation of the command-line progress bar defined in consoleProgressBar.hpp
 */


#include <iomanip>
#include <iostream>

#ifdef _WIN32
#include <Windows.h>
#include <Winbase.h>
#include <Wincon.h>
#elif __APPLE__
#include <sys/ioctl.h>
#include <unistd.h>
#elif __linux__
#include <sys/ioctl.h>
#include <unistd.h>
#endif

#include "consoleProgressBar.hpp"
#include "constants.hpp"


ConsoleProgressBar::ConsoleProgressBar() :
    progress(0), title(""), titleWidth(0)
{
    // Setup for printed output
    std::cout << std::fixed;
    std::cout << std::setprecision(2);

    calculateProgressBarWidth();
}


ConsoleProgressBar::ConsoleProgressBar(std::string title) :
    progress(0)
{
    // Setup for printed output
    std::cout << std::fixed;
    std::cout << std::setprecision(2);

    setTitle(title);
    calculateProgressBarWidth();
}


void ConsoleProgressBar::print(void)
{
    // Print out the title
    std::cout << title << std::string(titleWidth - title.length(), ' ') << ' ';

    // Print the progress bar and percentage
    std::cout << '[';
    auto pos = static_cast<uint8_t>(progressBarWidth * progress);
    std::cout << std::string(pos, '=') << std::string(progressBarWidth - pos, ' ');
    std::cout << "] " << std::setw(6) << min(100*progress, 100) << "%\r";
}


void ConsoleProgressBar::printProgress(double progress)
{
    setProgress(progress);
    print();
}


void ConsoleProgressBar::setProgress(double progress)
{
    this->progress = progress;
}



void ConsoleProgressBar::setTitle(std::string title)
{
    this->title = title;
    if (title.length() > titleWidth)
        setTitleWidth(title.length());
}


void ConsoleProgressBar::setTitleWidth(uint8_t titleWidth)
{
    this->titleWidth = titleWidth < title.length() ? title.length() : titleWidth;
    calculateProgressBarWidth();
}


void ConsoleProgressBar::calculateProgressBarWidth(void)
{
    // Determine the width of the console window
    progressBarWidth = DEFAULT_PBAR_WIDTH;
    #ifdef _WIN32
    CONSOLE_SCREEN_BUFFER_INFO csbi;
    if (GetConsoleScreenBufferInfo(GetStdHandle(static_cast<unsigned long>(-11)), &csbi))
        progressBarWidth = (csbi.srWindow.Right - csbi.srWindow.Left) - PERCENT_WIDTH;
    #elif __APPLE__
    struct winsize w;
    ioctl(STDOUT_FILENO, TIOCGWINSZ, &w);
    progressBarWidth = w.ws_col - PERCENT_WIDTH;
    #elif __linux__
    struct winsize w;
    ioctl(STDOUT_FILENO, TIOCGWINSZ, &w);
    progressBarWidth = w.ws_col - PERCENT_WIDTH;
    #endif

    progressBarWidth -= titleWidth + 1;
}