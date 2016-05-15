#include <cuda.h>
#include <cuda_runtime_api.h>
#include <stdio.h>
#include <thrust/remove.h>
#include <thrust/device_ptr.h>

#include "kernels/ID1.cu"
#include "kernels/ID2.cu"
#include "kernels/ID3.cu"
#include "kernels/ID4.cu"

#include "cudaCheckError.cu"

#define TH_PER_BLOCK 64
#define N_SCALES      1


void debugResults(int *facesDetected_d, float *results_d, int nValidSubWindows);

int compact(int *winOffsets_d, int *faceDetected_d, int nValidSubWindows);

void kernel_heading(char *heading, int blocks, int th_per_block, int threads, int nValidSubWindows);

void kernel_footer(char *msg, clock_t kernel_start);


void cuda_detect_faces(float *hostIntImage,
                       int rows,
                       int columns,
                       int stride,
                       int *hostSubWinOffsets,
                       int subWinNum,
                       int subWinSize)
{

   // Initialize kernel size --------------------------------------------------
   int blocksPerGrid = 1;
   int threadPerBlock = TH_PER_BLOCK;
   int threads = blocksPerGrid * threadPerBlock;


   // Initialize clock --------------------------------------------------------
   clock_t test_start = clock();
   clock_t kernel_start;


   // Copy Integral Image to device -------------------------------------------
   float *intImg_d;
   cudaMalloc(&intImg_d, rows * columns * sizeof(float));
   cudaMemcpy(intImg_d, hostIntImage, rows * columns * sizeof(float), cudaMemcpyHostToDevice);


   // Copy window mOffsets to device -------------------------------------------
   int *winOffsets_d;
   int nValidSubWindows = subWinNum;

   cudaMalloc(&winOffsets_d, nValidSubWindows * sizeof(int));
   cudaMemcpy(winOffsets_d, hostSubWinOffsets, nValidSubWindows * sizeof(int), cudaMemcpyHostToDevice);


   // Initialize device 'boolean' face detected array -------------------------
   int *faceDetected_d;
   cudaMalloc(&faceDetected_d, nValidSubWindows * sizeof(int));
   cudaMemset(faceDetected_d, 0, nValidSubWindows * sizeof(int));


   // Initialize results array for debugging... -------------------------------
   float *results_d;
   cudaMalloc(&results_d, nValidSubWindows * sizeof(float));
   cudaMemset(results_d, 0, nValidSubWindows * sizeof(float));



   //==========================================================================
   // Run ID1 -----------------------------------------------------------------
   kernel_heading("ID1", blocksPerGrid, threadPerBlock, threads, nValidSubWindows);
   kernel_start = clock();
   for (int i = 2; i < 2 + N_SCALES; ++i)
   {
      ID1kernel << < blocksPerGrid, threadPerBlock >> > (intImg_d,               // Itegral Image
              stride,                  //	Stride
              winOffsets_d,            //	Sub-Window Offsets
              subWinSize,               //	Sub-Window Size
              nValidSubWindows,      //	Number of Sub Windows
              subWinSize / (5 * (i)),         // Scale of the feature
              faceDetected_d,         //	Array to hold if a face was detected
              results_d               //	Array to hold maximum feature value for each sub window
      );
   }
   kernel_footer("ID1", kernel_start);
   debugResults(faceDetected_d, results_d, nValidSubWindows);

   // Compact -----------------------------------------------------------------
   nValidSubWindows = compact(winOffsets_d, faceDetected_d, nValidSubWindows);

   // Prepare for next run ----------------------------------------------------
   cudaMemset(faceDetected_d, 0, nValidSubWindows * sizeof(float));
   cudaMemset(results_d, 0, nValidSubWindows * sizeof(float));



   //==========================================================================
   // Run ID2 -----------------------------------------------------------------
   kernel_heading("ID2", blocksPerGrid, threadPerBlock, threads, nValidSubWindows);
   kernel_start = clock();
   for (int i = 2; i < 2 + N_SCALES; ++i)
   {
      ID2kernel << < blocksPerGrid, threadPerBlock >> > (intImg_d,               // Itegral Image
              stride,                  //	Stride
              winOffsets_d,            //	Sub-Window Offsets
              subWinSize,               //	Sub-Window Size
              nValidSubWindows,      //	Number of Sub Windows
              subWinSize / (5 * (i)),         // Scale of the feature
              faceDetected_d,         //	Array to hold if a face was detected
              results_d               //	Array to hold maximum feature value for each sub window
      );
   }
   kernel_footer("ID2", kernel_start);
   debugResults(faceDetected_d, results_d, nValidSubWindows);

   // Compact -----------------------------------------------------------------
   nValidSubWindows = compact(winOffsets_d, faceDetected_d, nValidSubWindows);

   // Prepare for next run ----------------------------------------------------
   cudaMemset(faceDetected_d, 0, nValidSubWindows * sizeof(float));
   cudaMemset(results_d, 0, nValidSubWindows * sizeof(float));



   //==========================================================================
   // Run ID3 -----------------------------------------------------------------
   kernel_heading("ID3", blocksPerGrid, threadPerBlock, threads, nValidSubWindows);
   kernel_start = clock();
   for (int i = 2; i < 2 + N_SCALES; ++i)
   {
      ID3kernel << < blocksPerGrid, threadPerBlock >> > (intImg_d,               // Itegral Image
              stride,                  //	Stride
              winOffsets_d,            //	Sub-Window Offsets
              subWinSize,               //	Sub-Window Size
              nValidSubWindows,      //	Number of Sub Windows
              subWinSize / (5 * (i)),         // Scale of the feature
              faceDetected_d,         //	Array to hold if a face was detected
              results_d               //	Array to hold maximum feature value for each sub window
      );
   }
   kernel_footer("ID3", kernel_start);
   debugResults(faceDetected_d, results_d, nValidSubWindows);

   // Compact -----------------------------------------------------------------
   nValidSubWindows = compact(winOffsets_d, faceDetected_d, nValidSubWindows);

   // Prepare for next run ----------------------------------------------------
   cudaMemset(faceDetected_d, 0, nValidSubWindows * sizeof(float));
   cudaMemset(results_d, 0, nValidSubWindows * sizeof(float));



   //==========================================================================
   // Run ID4 -----------------------------------------------------------------
   kernel_heading("ID4", blocksPerGrid, threadPerBlock, threads, nValidSubWindows);
   kernel_start = clock();
   for (int i = 2; i < 2 + N_SCALES; ++i)
   {
      ID4kernel << < blocksPerGrid, threadPerBlock >> > (intImg_d,               // Itegral Image
              stride,                  //	Stride
              winOffsets_d,            //	Sub-Window Offsets
              subWinSize,               //	Sub-Window Size
              nValidSubWindows,      //	Number of Sub Windows
              subWinSize / (5 * (i)),         // Scale of the feature
              faceDetected_d,         //	Array to hold if a face was detected
              results_d              //	Array to hold maximum feature value for each sub window
      );
   }
   kernel_footer("ID4", kernel_start);
   debugResults(faceDetected_d, results_d, nValidSubWindows);

   // Compact -----------------------------------------------------------------
   nValidSubWindows = compact(winOffsets_d, faceDetected_d, nValidSubWindows);


   //==========================================================================
   // Test Results ------------------------------------------------------------
   printf("Face Results\n\n");
   printf("Completed test in %f seconds\n", ((double) clock() - test_start) / CLOCKS_PER_SEC);
   if (nValidSubWindows > 0)
   {
      printf("A face was detected\n");
   }



   // Cleanup -----------------------------------------------------------------
   cudaFree(intImg_d);
   cudaFree(winOffsets_d);
   cudaFree(faceDetected_d);
   cudaFree(results_d);
}


void debugResults(int *facesDetected_d, float *results_d, int nValidSubWindows)
{
   int *facesDetected = (int *) malloc(nValidSubWindows * sizeof(int));
   float *results = (float *) malloc(nValidSubWindows * sizeof(float));

   cudaMemcpy(facesDetected, facesDetected_d, nValidSubWindows * sizeof(int), cudaMemcpyDeviceToHost);
   cudaMemcpy(results, results_d, nValidSubWindows * sizeof(float), cudaMemcpyDeviceToHost);


   for (int i = 0; i < nValidSubWindows; ++i)
   {
      printf("%4d - %f: ", i, results[i]);
      if (facesDetected[i] == 0)
      {
      } else if (facesDetected[i] == 1)
      {
         printf(" FACE DETECTED");
      } else
      {
         printf(" Well poo!!!");
      }
      printf("\n");
   }

   free(facesDetected);
   free(results);
}


int compact(int *winOffsets_d, int *faceDetected_d, int nValidSubWindows)
{
   clock_t clk = clock();

   // Cast to thrust device pointers
   thrust::device_ptr<int> offsets_ptr(winOffsets_d);
   thrust::device_ptr<int> detected_ptr(faceDetected_d);

   // Perform the compact!
   thrust::device_ptr<int> new_end = thrust::remove_if(offsets_ptr, offsets_ptr + nValidSubWindows, detected_ptr,
                                                       thrust::logical_not<int>());

   // Compute the length of compacted array
   int len = new_end - offsets_ptr;

   printf("Compacting completed in %f seconds\n", ((double) clock() - clk) / CLOCKS_PER_SEC);

   printf("Possible faces: %d\n\n", len);

   // Return the length of the compacted array
   return len;
}


void kernel_heading(char *heading, int blocks, int th_per_block, int threads, int nValidSubWindows)
{
   printf("\n");
   printf("Running %s --------\n", heading);
   printf("Blocks:   %d\n", blocks);
   printf("Th/Block: %d\n", th_per_block);
   printf("Threads:  %d\n", threads);
   printf("Windows:  %d\n", nValidSubWindows);
}

void kernel_footer(char *msg, clock_t kernel_start)
{
   cudaThreadSynchronize();
   printf("%s completed in %f seconds\n", msg, ((double) clock() - kernel_start) / CLOCKS_PER_SEC);
}