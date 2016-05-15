#include <cuda_runtime_api.h>
#include <device_launch_parameters.h>

#include <iostream>
#include <iomanip>

#include "IntegralImage.h"
#include "cudaCheckError.cu"

#define THREADS_PER_BLOCK 32

__global__
void horizontal_kernel(float *data, int rows, int cols, size_t stride)
{
   // start from row 0
   int row = blockIdx.x * blockDim.x + threadIdx.x;

   if (row < rows)
   {
      for (int col = 1; col < cols; ++col)
      {
         data[row * stride + col] = data[row * stride + col] + data[row * stride + col - 1];
      }
   }


}

__global__
void vertical_kernel(float *data, int rows, int cols, size_t stride)
{
   // Start from column 1
   int col = blockIdx.x * blockDim.x + threadIdx.x;

   if (col < cols)
   {
      for (int row = 1; row < rows; ++row)
      {
         data[row * stride + col] = data[row * stride + col] + data[(row - 1) * stride + col];
      }
   }
}


void integralImage(float *hostData, int rows, int cols, size_t stride)
{
   float *deviceData;
   cudaCheckError(cudaMalloc(&deviceData, rows * cols * sizeof(float)));
   cudaCheckError(cudaMemcpy(deviceData, hostData, rows * cols * sizeof(float), cudaMemcpyHostToDevice));

   int num_blocks = rows / THREADS_PER_BLOCK;

   cudaEvent_t start, end;
   cudaCheckError(cudaEventCreate(&start));
   cudaCheckError(cudaEventCreate(&end));

   cudaCheckError(cudaEventRecord(start, 0));
   horizontal_kernel <<< num_blocks, THREADS_PER_BLOCK >>> (deviceData, rows, cols, stride);

   num_blocks = cols / THREADS_PER_BLOCK + 1;

   cudaCheckError(cudaThreadSynchronize());


   vertical_kernel <<< num_blocks, THREADS_PER_BLOCK >>> (deviceData, rows, cols, stride);
   cudaCheckError(cudaThreadSynchronize());
   cudaCheckError(cudaEventRecord(end, 0));
   cudaCheckError(cudaEventSynchronize(end));

   float time = 0;
   cudaCheckError(cudaEventElapsedTime(&time, start, end));
   std::cout << "CUDA Integral Image: " << time << " ms" << std::endl;

   cudaCheckError(cudaMemcpy(hostData, deviceData, rows * cols * sizeof(float), cudaMemcpyDeviceToHost));

   cudaCheckError(cudaEventDestroy(start));
   cudaCheckError(cudaEventDestroy(end));
   cudaCheckError(cudaFree(deviceData));
}