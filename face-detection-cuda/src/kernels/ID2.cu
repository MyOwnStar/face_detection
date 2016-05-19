#include <device_launch_parameters.h>

#define BASE_WIDTH         8
#define BASE_HEIGHT         4
#define THRESHOLD         .08f   //definitely needs to be changed
#define SKIP_AMOUNT         4         //amount to skip in pixels, we can change this to be multiplied by scale if necessary/desirable

//This identifier is 2 horizontal bars with dark (negative) on top and light (positive) on bottom
__global__
void ID2kernel(float *intImage,         // Integral image
               int stride,               // Stride
               int *subWinOffsets,       // Sub-Window offsets
               int subWinSize,           // Sub-Window size
               int subWinNum,            // Number of Sub-Windows
               int scale,                // Scale of the feature
               int *faceDetected,        // Array to hold if a face was detected
               float *results)           // Array to hold maximum feature value for each sub-window
{
   int threadNum = blockIdx.x * blockDim.x + threadIdx.x;
   if (threadNum < subWinNum)
   {
      float maxFitValue = 0.0f;
      int startX = subWinOffsets[threadNum] / (stride);
      int startY = subWinOffsets[threadNum] % stride;
      for (int i = startX; (i + BASE_WIDTH * scale) < (startX + subWinSize); i = i + SKIP_AMOUNT)
      { //use SKIP_AMOUNT * scale for it to scale up as identifier scales
         for (int j = startY; (j + BASE_HEIGHT * scale) < (startY + subWinSize); j = j + SKIP_AMOUNT)
         {
            // take important corners from image
            float upperLeft = intImage[i * stride + j];
            float upperRight = intImage[(i + BASE_WIDTH * scale) * stride + j];
            float midLeft = intImage[i * stride + j + (BASE_HEIGHT * scale >> 1)];
            float midRight = intImage[(i + BASE_WIDTH * scale) * stride + j + (BASE_HEIGHT * scale >> 1)];
            float lowerLeft = intImage[i * stride + j + (BASE_HEIGHT * scale)];
            float lowerRight = intImage[(i + BASE_WIDTH * scale) * stride + j + (BASE_HEIGHT * scale)];

            //calulate fit value based on identifier (hard-coded)
            float fitValue = midLeft * 2 - midRight * 2 - upperLeft + lowerRight + upperRight - lowerLeft;

            if (fitValue > maxFitValue)
            {
               maxFitValue = fitValue;
            }
         }
      }

      // goodnessValue = fit/area
      float goodnessValue = maxFitValue / (BASE_WIDTH * scale * BASE_HEIGHT * scale);
      results[threadNum] = goodnessValue;

      if (goodnessValue > THRESHOLD)
      {
         faceDetected[threadNum] = 1;
      }
   }
}

