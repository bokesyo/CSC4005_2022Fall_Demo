// nvcc cuda-hello.cpp
#include <cstdio>
#include <cuda.h>
#include <cuda_runtime.h>

int main() {
  int deviceCount = 0;
  cudaError_t error_id = cudaGetDeviceCount(&deviceCount);
  printf("%d CUDA devices detected\n");
  return 0;
}
