#include <iostream>

#include "ImageProcessing.hpp"


int main(int argc, char** argv)
{
   cv::Mat srcImage;

   srcImage = getImage(argc, argv);
   std::cout << "Image resolution: " << srcImage.cols << "x" << srcImage.rows << std::endl;
   //display(srcImage, "Source Image");

   cv::Mat intImage;

   intImage = integralImage(srcImage);
   cv::normalize(intImage, intImage, 0, 1, CV_MINMAX);
   //display(intImage, "Integral Image");


   return 0;
}