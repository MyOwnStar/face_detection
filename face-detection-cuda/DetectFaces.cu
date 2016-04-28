#include "DetectFaces.hpp"

#include <thrust/remove.h>
#include <thrust/device_ptr.h>


#define NUM_SCALES 1

void debugResult(int *faceDetected_d, float *results_d, int nValidSubWindows);
void kernelInfo(std::string kernelID, int blocksPerGrid, int threadPerBlocks, int threads, int nValidSubWindows);
int compact(int *subWinOffsets_d, int *facesDetected_d, int nValidSubWindows);


void detectedFaces(float *intImage_h,
                 int rows,
                 int columns,
                 int stride,
                 int *subWinOffsets_h,
                 int subWinNum,
                 int subWinSize)
{
   // Kernel setup
   int blocksPerGrid = 1;
   int threadPerBlock = 64;
   int threads = blocksPerGrid * threadPerBlock;


   // Copying integral image to device
   float *intImage_d;
   // Size of integral image in bytes
   size_t intImageSize = rows * columns * sizeof(float);
   cudaCheckError(cudaMalloc(&intImage_d, intImageSize));
   cudaCheckError(cudaMemcpy(intImage_d, intImage_h, intImageSize, cudaMemcpyHostToDevice));


   // Copying window offsets to device
   int *subWinOffsets_d;
   int nValidSubWindows = subWinNum;
   // Size of sub-window offsets in bytes
   size_t subWinOffsetsSize = nValidSubWindows * sizeof(int);
   cudaCheckError(cudaMalloc(&subWinOffsets_d, subWinOffsetsSize));
   cudaCheckError(cudaMemcpy(subWinOffsets_d, subWinOffsets_h, subWinOffsetsSize, cudaMemcpyHostToDevice));


   // Init array for face detected,
   // containg 1 if face detected, else 0 - face not detected
   int *faceDetected_d;
   cudaCheckError(cudaMalloc(&faceDetected_d, nValidSubWindows * sizeof(int)));
   cudaCheckError(cudaMemset(faceDetected_d, 0, nValidSubWindows * sizeof(int)));


   // Array to hold maximum feature value for each sub window
   // for debugging
   float *results_d;
   cudaCheckError(cudaMalloc(&results_d, nValidSubWindows * sizeof(float)));
   cudaCheckError(cudaMemset(results_d, 0, nValidSubWindows * sizeof(float)));



   // Kernel #1
   kernelInfo("ID1", blocksPerGrid, threadPerBlock, threads, nValidSubWindows);
   for (int i = 2; i < 2 + NUM_SCALES; i++)
   {
      ID1Kernel<<<blocksPerGrid, threadPerBlock>>>(intImage_d,               // Integral image
                                                   stride,                   // Stride
                                                   subWinOffsets_d,          // Sub-window offsets
                                                   subWinSize,               // Sub-window size
                                                   nValidSubWindows,         // Number of sub-windows
                                                   subWinSize / (5 * i),     // Scale of the feature
                                                   faceDetected_d,           // Array to hold if face was detected
                                                   results_d);               // Array yo hold maximum feature value for each sub-window
   }
   cudaCheckError(cudaThreadSynchronize());

   debugResult(faceDetected_d, results_d, nValidSubWindows);

   nValidSubWindows = compact(subWinOffsets_d, faceDetected_d, nValidSubWindows);

   // Prepare for next run
   cudaCheckError(cudaMemset(faceDetected_d, 0, nValidSubWindows * sizeof(float)));
   cudaCheckError(cudaMemset(results_d, 0, nValidSubWindows * sizeof(float)));


   // Kernel #2
   kernelInfo("ID2", blocksPerGrid, threadPerBlock, threads, nValidSubWindows);
   for (int i = 2; i < 2 + NUM_SCALES; i++)
   {
      ID2Kernel<<<blocksPerGrid, threadPerBlock>>>(intImage_d,               // Integral image
                                                   stride,                   // Stride
                                                   subWinOffsets_d,          // Sub-window offsets
                                                   subWinSize,               // Sub-window size
                                                   nValidSubWindows,         // Number of sub-windows
                                                   subWinSize / (5 * i),     // Scale of the feature
                                                   faceDetected_d,           // Array to hold if face was detected
                                                   results_d);               // Array yo hold maximum feature value for each sub-window
   }
   cudaCheckError(cudaThreadSynchronize());

   debugResult(faceDetected_d, results_d, nValidSubWindows);

   nValidSubWindows = compact(subWinOffsets_d, faceDetected_d, nValidSubWindows);

   // Prepare for next run
   cudaCheckError(cudaMemset(faceDetected_d, 0, nValidSubWindows * sizeof(float)));
   cudaCheckError(cudaMemset(results_d, 0, nValidSubWindows * sizeof(float)));


   // Kernel 3
   kernelInfo("ID3", blocksPerGrid, threadPerBlock, threads, nValidSubWindows);
   for (int i = 2; i < 2 + NUM_SCALES; i++)
   {
      ID1Kernel<<<blocksPerGrid, threadPerBlock>>>(intImage_d,               // Integral image
                                                   stride,                   // Stride
                                                   subWinOffsets_d,          // Sub-window offsets
                                                   subWinSize,               // Sub-window size
                                                   nValidSubWindows,         // Number of sub-windows
                                                   subWinSize / (5 * i),     // Scale of the feature
                                                   faceDetected_d,           // Array to hold if face was detected
                                                   results_d);               // Array yo hold maximum feature value for each sub-window
   }
   cudaCheckError(cudaThreadSynchronize());

   debugResult(faceDetected_d, results_d, nValidSubWindows);

   nValidSubWindows = compact(subWinOffsets_d, faceDetected_d, nValidSubWindows);

   // Prepare for next run
   cudaCheckError(cudaMemset(faceDetected_d, 0, nValidSubWindows * sizeof(float)));
   cudaCheckError(cudaMemset(results_d, 0, nValidSubWindows * sizeof(float)));


   // Kernel 4
   kernelInfo("ID4", blocksPerGrid, threadPerBlock, threads, nValidSubWindows);
   for (int i = 2; i < 2 + NUM_SCALES; i++)
   {
      ID1Kernel<<<blocksPerGrid, threadPerBlock>>>(intImage_d,               // Integral image
                                                   stride,                   // Stride
                                                   subWinOffsets_d,          // Sub-window offsets
                                                   subWinSize,               // Sub-window size
                                                   nValidSubWindows,         // Number of sub-windows
                                                   subWinSize / (5 * i),     // Scale of the feature
                                                   faceDetected_d,           // Array to hold if face was detected
                                                   results_d);               // Array yo hold maximum feature value for each sub-window
   }
   cudaCheckError(cudaThreadSynchronize());

   debugResult(faceDetected_d, results_d, nValidSubWindows);

   nValidSubWindows = compact(subWinOffsets_d, faceDetected_d, nValidSubWindows);

   // Results
   if (nValidSubWindows > 0)
   {
      std::cout << "A face was detected" << std::endl;
   }

   cudaFree(intImage_d);
   cudaFree(subWinOffsets_d);
   cudaFree(faceDetected_d);
   cudaFree(results_d);
}

void kernelInfo(std::string kernelID, int blocksPerGrid, int threadPerBlock, int threads, int nValidSubWindows)
{
   std::cout << "Running " << kernelID << std::endl;
   std::cout << "Blocks per grid: " << blocksPerGrid << std::endl;
   std::cout << "Threads per blocks " << threadPerBlock << std::endl;
   std::cout << "Threads: " <<  threads <<std::endl;
   std::cout << "Sub-windows: " << nValidSubWindows << std::endl;
}

void compact(int *subWinOffsets_d, int *facesDetected_d, int nValidSubWindows)
{
   // Cast to thrust device pointers
   thrust::device_ptr<int> offsets_ptr(subWinOffsets_d);
   thrust::device_ptr<int> detected_ptr(faceDetected_d);

   // Perform the compact!
   thrust::device_ptr<int> new_end = thrust::remove_if(offsets_ptr, offsets_ptr + nValidSubWindows, detected_ptr, thrust::logical_not<int>());

   // Compute the length of compacted array
   int len = new_end - offsets_ptr;

   printf("Possible faces: %d\n\n", len);

   // Return the length of the compacted array
   return len;
}

void debugResult(int *faceDetected_d, float *results_d, int nValidSubWindows)
{
   int *facesDetected_h = new int[nValidSubWindows * sizeof(int)];
   float *results_h = new float[nValidSubWindows * sizeof(float)];

   cudaCheckError(cudaMemCpy(facesDetected_h, faceDetected_d, nValidSubWindows * sizeof(int), cudaMemcpyDeviceToHost));
   cudaCheckError(cudaMemCpy(results_h, results_d, nValidSubWindows * sizeof(float), cudaMemcpyDeviceToHost));

   for (int i = 0; i < nValidSubWindows; i++)
   {
      std::cout << i << " - " << results_h[i];
      if (facesDetected_h[i] == 1)
      {
         std::cout << "Face detected!";
      }
      else
      {
         std::cout << "Face not detected!";
      }
   }

   delete[] facesDetected_h;
   delete[] results_h;
}
