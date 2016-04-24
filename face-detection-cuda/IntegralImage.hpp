#ifndef INTEGRALIMAGE_HPP
#define INTEGRALIMAGE_HPP

#include "cudaCheckError.hpp"
#include "cudaInclude.hpp"

#include <iostream>

void gpuIntImage(float *hostData, int rows, int columns, size_t stride);


#endif // INTEGRALIMAGE_HPP