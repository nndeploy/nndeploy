
// #include "acl/acl.h"
// #include "kernel_operator.h"

// extern "C" __global__ __aicore__ void hello_world() {
//   AscendC::printf("Hello World!!!\n");
// }

// void hello_world_do(uint32_t blockDim, void *stream) {
//   hello_world<<<blockDim, nullptr, stream>>>();
// }

// int32_t main(int argc, char const *argv[]) {
//   aclInit(nullptr);
//   int32_t deviceId = 0;
//   aclrtSetDevice(deviceId);
//   aclrtStream stream = nullptr;
//   aclrtCreateStream(&stream);

//   constexpr uint32_t blockDim = 8;
//   hello_world_do(blockDim, stream);
//   aclrtSynchronizeStream(stream);

//   aclrtDestroyStream(stream);
//   aclrtResetDevice(deviceId);
//   aclFinalize();
//   return 0;
// }