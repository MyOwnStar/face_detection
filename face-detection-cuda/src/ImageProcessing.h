//
// Created by alex on 14.05.16.
//

#ifndef EEC_277_GPU_FACE_DETECT_IMAGEPROCESSING_H
#define EEC_277_GPU_FACE_DETECT_IMAGEPROCESSING_H

#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

#include <iostream>
#include <iomanip>
#include <chrono>
#include <string>
#include <ctime>

#include "IntegralImage.h"
#include "detectFaces.h"
#include "WindowInfo.h"


// Load image and convert to grayscale
cv::Mat getImage(int argc, char **argv);

// Calculate intImage on cpu
cv::Mat integralImageCPU(cv::Mat &image);

// Calculate intImage on gpu
cv::Mat integralImageGPU(cv::Mat &cudaIntImage);

// display image
void display(cv::Mat &image, std::string title);

// main function for detection faces
void detectFaces(cv::Mat &integralImage);

#endif //EEC_277_GPU_FACE_DETECT_IMAGEPROCESSING_H
