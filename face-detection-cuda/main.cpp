#include <iostream>

#include "ImageProcessing.hpp"

int main(int argc, char** argv)
{
   cv::Mat srcImage;

   srcImage = getImage(argc, argv);
   std::cout << "Image resolution: " << srcImage.cols << "x" << srcImage.rows << std::endl;
   //display(srcImage, "Source Image");

   
   cv::Mat intImageCPU;

   intImageCPU = integralImageCPU(srcImage);
   cv::normalize(intImageCPU, intImageCPU, 0, 1, CV_MINMAX);
   display(intImageCPU, "OpenCV Integral image");


   cv::Mat intImageGPU;

   intImageGPU = integralImageGPU(srcImage);
   cv::normalize(intImageGPU, intImageGPU, 0, 1, CV_MINMAX);
   display(intImageGPU, "CUDA Integral image");

   return 0;
}