#include "DetectFaces.hpp"


#define NUM_SCALES 1


void detectFaces(float *intImage_h,
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
   cudaCheckError(cudaMemset(faceDetected_d, 0, nValidSubWindows * sizeof(int));


   // Array to hold maximum feature value for each sub window
   // for debugging
   float *results_d;
   cudaCheckError(cudaMalloc(&results_d, nValidSubWindows * sizeof(float)));
   cudaCheckError(cudaMemset(results_d, 0, nValidSubWindows * sizeof(float)));



}

