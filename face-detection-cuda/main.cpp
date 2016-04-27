#include <iostream>

#include "ImageProcessing.hpp"
#include "WinInfo.hpp"

int main(int argc, char** argv)
{
   cv::Mat srcImage;

   srcImage = getImage(argc, argv);

   //display(srcImage, "Source Image");

   // Calculate sub-window size
   int subWinSize = std::min(srcImage.rows, srcImage.cols) / 4;

   WinInfo winInfo(srcImage, subWinSize);

   cv::Mat intImageCPU;

   intImageCPU = integralImageCPU(srcImage);
   cv::normalize(intImageCPU, intImageCPU, 0, 1, CV_MINMAX);
   //display(intImageCPU, "OpenCV Integral image");


   cv::Mat intImageGPU;

   intImageGPU = integralImageGPU(srcImage);
   cv::normalize(intImageGPU, intImageGPU, 0, 1, CV_MINMAX);
   //display(intImageGPU, "CUDA Integral image");

   return 0;
}