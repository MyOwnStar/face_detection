#ifndef _CUDA_DETECT_FACES_
#define _CUDA_DETECT_FACES_


void cuda_detect_faces(float* hostIntImage, int rows, int columns, int stride, int* hostSubWinOffsets, int subWinNum, int subWinSize);

#endif // _CUDA_DETECT_FACES_