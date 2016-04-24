cmake_minimum_required(VERSION 2.8)
project(face_detection_cuda)

find_package(CUDA REQUIRED)
find_package(OpenCV REQUIRED)

file(GLOB SOURCES "*.cpp" "*.hpp" "*.cu" "Kernels/*")


cuda_add_executable(face_detection_cuda ${SOURCES})

list(APPEND CMAKE_CXX_FLAGS "-std=c++11 -Wall")

list(APPEND CUDA_NVCC_FLAGS --compiler-options -fno-strict-aliasing -lineinfo -use_fast_math -Xptxas -dlcm=cg)
list(APPEND CUDA_NVCC_FLAGS -gencode arch=compute_11,code=sm_11)

target_link_libraries(face_detection_cuda ${OpenCV_LIBS})