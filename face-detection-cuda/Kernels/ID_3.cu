#include "../cudaInclude.hpp"

//This identifier is 3 vertical bars going dark light dark

#define ID4_BASE_WIDTH		3
#define ID4_BASE_HEIGHT		6
#define ID4_MID_WIDTH		1
#define ID4_THRESHOLD		.19f
#define ID4_SKIP_AMOUNT		1


__global__
void ID3Kernel(float  *intImage,             // Integral image
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

      for (int i = x; (i + ID4_BASE_WIDTH * scale) < (x + subWinSize); i += ID4_SKIP_AMOUNT)
      {
         for (int j = y; (j + ID4_BASE_HEIGHT * scale) < (y + subWinSize); j += ID4_SKIP_AMOUNT)
         {
            float upperLeft = intImage[i * stride + j];
            float upperRight = intImage[(i + ID4_BASE_WIDTH * scale) * stride + j];

            float midLeftTop = intImage[(i + ID4_BASE_WIDTH * scale / 2 - ID4_MID_WIDTH * scale / 2) * stride + j];
            float midRightTop	= intImage[(i + ID4_BASE_WIDTH * scale / 2 + ID4_MID_WIDTH * scale / 2) * stride + j];
            float midLeftBot = intImage[(i + ID4_BASE_WIDTH * scale / 2 - ID4_MID_WIDTH * scale / 2) * stride + j + ID4_BASE_HEIGHT * scale];
            float midRightBot	= intImage[( + ID4_BASE_WIDTH * scale / 2 + ID4_MID_WIDTH * scale / 2) * stride + j + ID4_BASE_HEIGHT*scale];

            float lowerLeft = intImage[i * stride + j + (ID4_BASE_HEIGHT * scale)];
            float lowerRight = intImage[(i + ID4_BASE_WIDTH * scale) * stride + j + (ID4_BASE_HEIGHT * scale)];

            //calulate fit value based on identifier
            float fitValue = (float) (2.0 * (midRightBot - midLeftBot - midRightTop + midLeftTop) -
                                     (lowerRight - lowerLeft - upperRight + upperLeft));

            if(fitValue < 0)
            {
               fitValue = -fitValue;
            }

            if(fitValue > maxFitValue)
            {
               maxFitValue = fitValue;
            }
         }
      }

      // goodnessValue = fit / area
      float goodnessValue = maxFitValue / (ID4_BASE_WIDTH * scale * ID4_BASE_HEIGHT * scale);
      results[threadNum] = goodnessValue;

      if(goodnessValue > ID4_THRESHOLD)
      {
         faceDetected[threadNum] = 1;
      }
   }
}