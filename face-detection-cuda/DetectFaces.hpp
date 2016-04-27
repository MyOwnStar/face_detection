#ifndef DETECTFACES_HPP
#define DETECTFACES_HPP

#include "cudaInclude.hpp"
#include "cudaCheckError.hpp"


void detectFaces(float *intImage_h,
                 int rows,
                 int columns,
                 int stride,
                 int *subWinOffsets_h,
                 int subWinNum,
                 int subWinSize);



#endif // DETECTFACES_HPP