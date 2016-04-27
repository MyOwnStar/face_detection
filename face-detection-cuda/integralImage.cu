#include "IntegralImage.hpp"

#define THREADS_PER_BLOCK 384

__global__
void rowsScan(float *data, int rows, int columns, size_t stride)
{
   int row = blockIdx.x * blockDim.x + threadIdx.x;

   if(row < rows)
   {
      for(int col = 1; col < columns; ++col)
      {
         data[row * stride + col] = data[row * stride + col] + data[row * stride + col - 1];
      }
   }
}

__global__
void colsScan(float *data, int rows, int columns, size_t stride)
{
   int col = blockIdx.x * blockDim.x + threadIdx.x;

   if(col < columns)
   {
      for(int row = 1; row < rows; ++row)
      {
         data[row * stride + col] = data[row * stride + col] + data[(row - 1) * stride + col];
      }
   }
}

void gpuIntImage(float *hostData, int rows, int columns, size_t stride)
{
   size_t sizeInByte = rows * columns * sizeof(float);

   float *deviceData;

   cudaCheckError(cudaMalloc(&deviceData, sizeInByte));

   cudaCheckError(cudaMemcpy(deviceData, hostData, sizeInByte, cudaMemcpyHostToDevice));

   int numBlocks = rows / THREADS_PER_BLOCK;

   //std::cout << "numBlocks: " << numBlocks << std::endl;

   const clock_t start = clock();
   rowsScan<<<numBlocks, THREADS_PER_BLOCK>>>(deviceData, rows, columns, stride);
   cudaCheckError(cudaPeekAtLastError());
   cudaCheckError(cudaDeviceSynchronize());

   numBlocks = columns / THREADS_PER_BLOCK + 1;

   //std::cout << "numBlocks: " << numBlocks << std::endl;

   cudaCheckError(cudaThreadSynchronize());

   colsScan<<<numBlocks, THREADS_PER_BLOCK>>>(deviceData, rows, columns, stride);
   cudaCheckError(cudaPeekAtLastError());
   cudaCheckError(cudaDeviceSynchronize());

   cudaCheckError(cudaThreadSynchronize());
   std::cout << "CUDA Intagral Image: " << static_cast<float>(clock() - start) << " ms" << std::endl;

   cudaCheckError(cudaMemcpy(hostData, deviceData, sizeInByte, cudaMemcpyDeviceToHost));

   cudaCheckError(cudaFree(deviceData));
}