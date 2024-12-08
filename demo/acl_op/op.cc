#include "kernel_operator.h"
extern "C" __global__ __aicore__ void hello_world() {
  AscendC::printf("Hello World!!!\n");
}

void hello_world_do(uint32_t blockDim, void* stream) {
  hello_world<<<blockDim, nullptr, stream>>>();
}