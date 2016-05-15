#include <device_launch_parameters.h>

#define ID1_BASE_WIDTH         8
#define ID1_BASE_HEIGHT         4
#define ID1_THRESHOLD         .05f   //definitely needs to be changed
#define ID1_SKIP_AMOUNT         4    //amount to skip in pixels, we can change this to be multiplied by scale if necessary/desirable

//This identifier is 2 horizontal bars with light (positive) on top and dark (negative) on bottom
__global__
void ID1kernel(float *intImage, size_t stride, int *offsets, int windowSize, int numSubWindows, int scale,
               int *faceDetected, float *results)
{
   int threadNum = blockIdx.x * blockDim.x + threadIdx.x;
   if (threadNum < numSubWindows)
   {
      float maxFitValue = 0.0f;
      int startX = offsets[threadNum] / (stride);
      int startY = offsets[threadNum] % stride;
      for (int i = startX; (i + ID1_BASE_WIDTH * scale) < (startX + windowSize); i = i + ID1_SKIP_AMOUNT)
      { //use ID1_SKIP_AMOUNT * scale for it to scale up as identifier scales
         for (int j = startY; (j + ID1_BASE_HEIGHT * scale) < (startY + windowSize); j = j + ID1_SKIP_AMOUNT)
         {
            // take important corners from image
            float upperLeft = intImage[i * stride + j];
            float upperRight = intImage[(i + ID1_BASE_WIDTH * scale) * stride + j];
            float midLeft = intImage[i * stride + j + (ID1_BASE_HEIGHT * scale >> 1)];
            float midRight = intImage[(i + ID1_BASE_WIDTH * scale) * stride + j + (ID1_BASE_HEIGHT * scale >> 1)];
            float lowerLeft = intImage[i * stride + j + (ID1_BASE_HEIGHT * scale)];
            float lowerRight = intImage[(i + ID1_BASE_WIDTH * scale) * stride + j + (ID1_BASE_HEIGHT * scale)];

            //calulate fit value based on identifier (hard-coded)
            float fitValue = midRight * 2 - midLeft * 2 + upperLeft - lowerRight - upperRight + lowerLeft;

            if (fitValue > maxFitValue)
            {
               maxFitValue = fitValue;
            }
         }
      }
      // goodnessValue = fit/area
      float goodnessValue = maxFitValue / (ID1_BASE_WIDTH * scale * ID1_BASE_HEIGHT * scale);
      results[threadNum] = goodnessValue;

      if (goodnessValue > ID1_THRESHOLD)
      {
         faceDetected[threadNum] = 1;
      }
   }
}

