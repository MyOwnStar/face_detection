#ifndef WININFO_HPP
#define WININFO_HPP

#include <opencv2/core.hpp>

#include <iostream>

class WinInfo
{
public:
   WinInfo(const cv::Mat image, int subWinSize);
   ~WinInfo();

   int xWindows();
   int yWindows();

   int xOffsetWindows();
   int yOffsetWindows();
   int xyOffsetWindows();

   int totalWindows();
   int subWindowSize();
   int windowOffset();
   int *subWindowOffsets();

private:
   cv::Mat _image;
   int _subWinSize;
   int _offset;
   int *_offsets;

   void computePositions();
};


#endif // WININFO_HPP
