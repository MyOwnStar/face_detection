#include "WindowInfo.h"


WindowInfo::WindowInfo(cv::Mat image, int subWindowSize)
{
   mImage = image;
   std::cout << "Image Size " << ": (" << mImage.rows << " , " << mImage.cols << ")" << std::endl;

   mSubWindowSize = subWindowSize;
   mOffset = mSubWindowSize / 2;
   mOffsets = new int[totalWindows()];
   computePositions();
}

WindowInfo::~WindowInfo()
{
   delete[] mOffsets;
}

int WindowInfo::xWindows()
{
   return mImage.cols / subWindowSize();
}

int WindowInfo::yWindows()
{
   return mImage.rows / subWindowSize();
}

int WindowInfo::xOffsetWindows()
{
   return (mImage.cols + mOffset) / subWindowSize() - 1;
}

int WindowInfo::yOffsetWindows()
{
   return (mImage.rows + mOffset) / subWindowSize() - 1;
}

int WindowInfo::xyOffsetWindows()
{
   return xOffsetWindows() * yOffsetWindows();
}

int WindowInfo::totalWindows()
{
   return xWindows() * yWindows()
          + xOffsetWindows() * yWindows()
          + xWindows() * yOffsetWindows()
          + xyOffsetWindows();
}

int WindowInfo::subWindowSize()
{
   return mSubWindowSize;
}

int WindowInfo::windowOffset()
{
   return mOffset;
}

int *WindowInfo::subWindowOffsets()
{
   return mOffsets;
}

void WindowInfo::computePositions()
{
   int win = 0;
   for (int i = 0; i <= mImage.rows - subWindowSize(); i += windowOffset())
   {
      for (int j = 0; j <= mImage.cols - subWindowSize(); j += windowOffset())
      {
         if (win < totalWindows())
         {
            mOffsets[win] = i * mImage.cols + j;
         }
         win++;
      }
   }
}