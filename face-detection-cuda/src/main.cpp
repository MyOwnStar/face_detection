#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>

#include "ImageProcessing.h"


int main(int argc, char **argv)
{
   cv::Mat sourceImageCPU;
   sourceImageCPU = getImage(argc, argv);

   cv::Mat intImgCPU;
   intImgCPU = integralImageCPU(sourceImageCPU);
   cv::normalize(intImgCPU, intImgCPU, 0, 1, CV_MINMAX);
   //display(intImgCPU, "OpenCV Integral Image");


   cv::Mat sourceImageGPU;
   sourceImageGPU = getImage(argc, argv);

   cv::Mat intImgGPU;
   intImgGPU = integralImageGPU(sourceImageGPU);
   cv::normalize(intImgGPU, intImgGPU, 0, 1, CV_MINMAX);
   //display(intImgGPU, "CUDA Integral Image");


   //detectFaces(intImgGPU);

   return 0;
}
