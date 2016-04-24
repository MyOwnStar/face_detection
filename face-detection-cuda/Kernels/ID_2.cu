#include "../cudaInclude.hpp"

//This identifier is 2 horizontal bars with dark on top and light on bottom

#define ID2_BASE_WIDTH			8
#define ID2_BASE_HEIGHT			4
#define ID2_THRESHOLD			.05f
#define ID2_SKIP_AMOUNT			4

__global__
void ID2Kernel(float  *intImage,             // Integral image
               int     stride,               // Stride
               int    *subWinOffset,         // Sub-Window offsets
               int     subWinSize,           // Sub-Window size
               int     subWinNum,            // Number of Sub-Windows
               int     scale,                // Scale of the feature
               int    *faceDetected,         // Array to hold if a face was detected
               float  *results)              // Array to hold maximum feature value for each sub-window
{
   int threadNum = blockIdx.x * blockDim.x + threadIdx.x;

   if (threadNum < subWinNum)
   {
      float maxFitValue = 0.0f;
      int x = subWinOffset[threadNum] / stride;
      int y = subWinOffset[threadNum] % stride;

      for (int i = x; (i + ID2_BASE_WIDTH * scale) < (x + subWinSize); i += ID2_SKIP_AMOUNT)
      {
         for (int j = y; (j + ID2_BASE_HEIGHT * scale) < (y + subWinSize); j += ID2_SKIP_AMOUNT)
         {
            float upperLeft = intImage[i * stride + j];
            float upperRight = intImage[(i + ID2_BASE_WIDTH * scale) * stride + j];

            float midLeft = intImage[i * stride + j + (ID2_BASE_HEIGHT * scale >> 1)];
            float midRight = intImage[(i + ID2_BASE_WIDTH * scale) * stride + j + (ID2_BASE_HEIGHT * scale >> 1)];

            float lowerLeft = intImage[i * stride + j + (ID2_BASE_HEIGHT * scale)];
            float lowerRight = intImage[(i + ID2_BASE_WIDTH * scale) * stride + j + (ID2_BASE_HEIGHT * scale)];

            //calulate fit value based on identifier
            float fitValue = midRight * 2 - midLeft * 2 + upperLeft - lowerRight - upperRight + lowerLeft;

            if(fitValue > maxFitValue)
            {
               maxFitValue = fitValue;
            }
         }
      }

      // goodnessValue = fit / area
      float goodnessValue = maxFitValue / (ID2_BASE_WIDTH * scale * ID2_BASE_HEIGHT * scale);
      results[threadNum] = goodnessValue;

      if(goodnessValue > ID2_THRESHOLD)
      {
         faceDetected[threadNum] = 1;
      }
   }
}