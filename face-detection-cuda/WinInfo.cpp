#include "WinInfo.hpp"

WinInfo::WinInfo(const cv::Mat image, int subWinSize)
{
   _image = image;
   _subWinSize = subWinSize;

   std::cout << "Image resolution: " << _image.cols << "x" << _image.rows << std::endl;
   std::cout << "Sub-window size : " << _subWinSize << std::endl;


   _offset = _subWinSize / 2;
   _offsets = new int [totalWindows()];

   computePositions();
}

WinInfo::~WinInfo()
{
   delete[] _offsets;
}

int WinInfo::xWindows()
{
    return _image.cols / windowSize();
}

int WinInfo::yWindows()
{
    return _image.rows / windowSize();
}

int WinInfo::xOffsetWindows()
{
    return (_image.cols + _offset) / windowSize() - 1;
}

int WinInfo::yOffsetWindows()
{
    return (_image.rows + _offset) / windowSize() - 1;
}

int WinInfo::xyOffsetWindows()
{
    return xOffsetWindows() + yOffsetWindows();
}

int WinInfo::totalWindows()
{
    return xWindows() * yWindows() + xOffsetWindows() * yWindows() + xWindows() * yOffsetWindows() + xyOffsetWindows();
}

int WinInfo::windowSize()
{
    return _subWinSize;
}

int WinInfo::windowOffset()
{
    return _offset;
}

int *WinInfo::subWindowOffsets()
{
    return _offsets;
}

void WinInfo::computePositions()
{
    int win = 0;
    for (int i = 0; i <= _image.rows - windowSize(); i += windowOffset())
    {
        for (int j = 0; j <= _image.cols - windowSize(); j += windowOffset())
        {
            if (win < totalWindows())
            {
                _offsets[win] = i * _image.cols + j;
            }
            win++;
        }
    }
}