#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

#include <iostream>
#include <string>
#include <ctime>

// Load image and convert to grayscale
cv::Mat getImage(int argc, char **argv);

// Calculate intImage on cpu and gpu
cv::Mat integralImage(cv::Mat &image);

// display image
void display(cv::Mat &image, std::string title);



cv::Mat getImage(int argc, char **argv)
{
   if (argc < 2)
   {
      std::cout << "Usage: ./detect <image>" << std::endl;
      exit(-1);
   }

   cv::Mat image = cv::imread(argv[1], CV_LOAD_IMAGE_GRAYSCALE);

   if (!image.data)
   {
      std::cout << "Could not open of find image" << std::endl;
      exit(-1);
   }

   return image;
}

cv::Mat integralImage(cv::Mat &image)
{
   image.convertTo(image, CV_32F, 1.0 / 255);

   cv::Mat opencvIntImage(image.rows + 1, image.cols + 1, CV_32F, 0.0f);

   const clock_t start = clock();
   cv::integral(image, opencvIntImage, CV_32F);
   std::cout << "OpenCV Intagral Image: " << static_cast<float>(clock() - start) / CLOCKS_PER_SEC << std::endl;


   return opencvIntImage;
}

void display(cv::Mat &image, std::string title)
{
   cv::namedWindow(title, CV_WINDOW_AUTOSIZE);
   cv::imshow(title, image);

   cv::waitKey(0);
}


