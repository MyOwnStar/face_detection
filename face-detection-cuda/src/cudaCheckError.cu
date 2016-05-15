#ifndef CUDACHECKERROR_HPP
#define CUDACHECKERROR_HPP

#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>


#define cudaCheckError(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
    if (code != cudaSuccess)
    {
        fprintf(stderr,"GPUError: %s %s %d\n", cudaGetErrorString(code), file, line);
        if (abort) exit(code);
    }
}


// Usage:
//          gpuErrchk( cudaMalloc(&a_d, size*sizeof(int)) );
//
// For kernels:
//
//          kernel<<<1,1>>>(a);
//          gpuErrchk( cudaPeekAtLastError() );
//          gpuErrchk( cudaDeviceSynchronize() );

#endif // CUDACHECKERROR_HPP