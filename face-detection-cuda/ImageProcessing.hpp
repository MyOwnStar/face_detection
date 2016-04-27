#ifndef IMAGEPROCESSING_HPP
#define IMAGEPROCESSING_HPP

#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

#include <iostream>
#include <string>
#include <ctime>

#include "IntegralImage.hpp"

// Load image and convert to grayscale
cv::Mat getImage(int argc, char **argv);

// Calculate intImage on cpu
cv::Mat integralImageCPU(cv::Mat &image);

// Calculate intImage on gpu
cv::Mat integralImageGPU(cv::Mat &cudaIntImage);

// display image
void display(cv::Mat &image, std::string title);



cv::Mat getImage(int argc, char **argv)
{
//   if (argc < 2)
//   {
//      std::cout << "Usage: ./detect <image>" << std::endl;
//      exit(-1);
//   }

   //cv::Mat image = cv::imread(argv[1], CV_LOAD_IMAGE_GRAYSCALE);
   cv::Mat image = cv::imread("/home/alex/Projects/face-detection-cuda/build/Data/1024x768/1.jpg", CV_LOAD_IMAGE_GRAYSCALE);

   if (!image.data)
   {
      std::cout << "Could not open of find image" << std::endl;
      exit(-1);
   }

   return image;
}

cv::Mat integralImageCPU(cv::Mat &image)
{
   image.convertTo(image, CV_32F, 1.0 / 255);

   cv::Mat opencvIntImage(image.rows + 1, image.cols + 1, CV_32F, 0.0f);

   const clock_t start = clock();
   cv::integral(image, opencvIntImage, CV_32F);
   std::cout << "OpenCV Intagral Image: " << static_cast<float>(clock() - start) << " ms" << std::endl;

   return opencvIntImage;
}

cv::Mat integralImageGPU(cv::Mat &image)
{
   image.convertTo(image, CV_32F, 1.0 / 255);

   cv::Mat cudaIntImage(image.rows + 1, image.cols + 1, CV_32F, 0.0f);
   cv::Mat mask(cudaIntImage, cv::Range(1, cudaIntImage.rows), cv::Range(1, cudaIntImage.cols));

   image.copyTo(mask);

   gpuIntImage((float*)cudaIntImage.data, cudaIntImage.rows, cudaIntImage.cols, cudaIntImage.step[0] / cudaIntImage.step[1]);

   return cudaIntImage;
}

void display(cv::Mat &image, std::string title)
{
   cv::namedWindow(title, CV_WINDOW_NORMAL);
   cv::imshow(title, image);

   cv::waitKey(0);
}


#endif // IMAGEPROCESSING_HPP