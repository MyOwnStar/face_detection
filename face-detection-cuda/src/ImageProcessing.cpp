#include "ImageProcessing.h"

cv::Mat getImage(int argc, char **argv)
{
   if (argc < 2)
   {
      std::cout << "Usage: ./detect <image>" << std::endl;
      exit(-1);
   }

   cv::Mat image = cv::imread(argv[1], CV_LOAD_IMAGE_GRAYSCALE);
   //cv::Mat image = cv::imread("/home/alex/Projects/face-detection-cuda/build/test.jpg", CV_LOAD_IMAGE_GRAYSCALE);

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

   auto start = std::chrono::steady_clock::now();
   cv::integral(image, opencvIntImage, CV_32F);
   auto end = std::chrono::steady_clock::now();
   std::cout << std::setprecision(2) << "OpenCV Integral Image: " << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count()
             << " ms" << std::endl;

   return opencvIntImage;
}

cv::Mat integralImageGPU(cv::Mat &image)
{
   image.convertTo(image, CV_32F, 1.0 / 255);

   cv::Mat cudaIntImage(image.rows + 1, image.cols + 1, CV_32F, 0.0f);
   cv::Mat mask(cudaIntImage, cv::Range(1, cudaIntImage.rows), cv::Range(1, cudaIntImage.cols));

   image.copyTo(mask);

   integralImage((float *) cudaIntImage.data, cudaIntImage.rows, cudaIntImage.cols,
                 cudaIntImage.step[0] / cudaIntImage.step[1]);

   return cudaIntImage;
}

void display(cv::Mat &image, std::string title)
{
   cv::namedWindow(title);
   cv::imshow(title, image);

   cv::waitKey(0);
}

void detectFaces(cv::Mat &integralImage)
{
   int winSize = std::min(integralImage.rows, integralImage.cols) / 4;
   std::cout << "Sub-Window Size: " << winSize << std::endl;

   WindowInfo winInfo(integralImage, winSize);

   std::cout << std::endl << std::endl << "CASCADING" << std::endl;

   cuda_detect_faces((float *)integralImage.data,
                     integralImage.rows,
                     integralImage.cols,
                     integralImage.cols,
                     winInfo.subWindowOffsets(),
                     winInfo.totalWindows(),
                     winInfo.subWindowSize()
                     );
}