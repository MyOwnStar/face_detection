#include <device_launch_parameters.h>

#define BASE_WIDTH      3
#define BASE_HEIGHT      6
#define BASE_MID_WIDTH      1
#define THRESHOLD      .19f   //definitely needs to be changed
#define SKIP_AMOUNT      1      //amount to skip in pixels, we can change this to be multiplied by scale if necessary/desirable


//This identifier is 3 vertical bars going dark light dark
__global__
void ID3kernel(float *intImage,         // Integral image
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
      int startX = subWinOffsets[threadNum] / (stride);
      int startY = subWinOffsets[threadNum] % stride;
      float maxFitValue = 0.0f;

      for (int i = startX; (i + BASE_WIDTH * scale) < (startX + subWinSize); i = i + SKIP_AMOUNT)
      { //use SKIP_AMOUNT * scale for it to scale up as identifier scales
         for (int j = startY; (j + BASE_HEIGHT * scale) < (startY + subWinSize); j = j + SKIP_AMOUNT)
         {
            // take important corners from image
            float upperLeft = intImage[i * stride + j];
            float upperRight = intImage[(i + BASE_WIDTH * scale) * stride + j];

            float midLeftTop = intImage[(i + BASE_WIDTH * scale / 2 - BASE_MID_WIDTH * scale / 2) * stride + j];
            float midRightTop = intImage[(i + BASE_WIDTH * scale / 2 + BASE_MID_WIDTH * scale / 2) * stride + j];
            float midLeftBot = intImage[(i + BASE_WIDTH * scale / 2 - BASE_MID_WIDTH * scale / 2) * stride + j +
                                        BASE_HEIGHT * scale];
            float midRightBot = intImage[(i + BASE_WIDTH * scale / 2 + BASE_MID_WIDTH * scale / 2) * stride + j +
                                         BASE_HEIGHT * scale];

            float lowerLeft = intImage[i * stride + j + (BASE_HEIGHT * scale)];
            float lowerRight = intImage[(i + BASE_WIDTH * scale) * stride + j + (BASE_HEIGHT * scale)];

            //calculate fit value based on identifier (hard-coded)
            float fitValue = 2.0 * (midRightBot - midLeftBot - midRightTop + midLeftTop) -
                             (lowerRight - lowerLeft - upperRight + upperLeft);

            if (fitValue < 0)
               fitValue = -fitValue;

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
