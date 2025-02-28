#ifndef ASCENDC_CPU_DEBUG
#include "acl/acl.h"
#include "aclrtlaunch_add_custom.h"
// extern void add_custom_do(uint32_t blockDim, void *stream, uint8_t *x,
//                           uint8_t *y, uint8_t *z);
#else
#include "tikicpulib.h"
extern "C" __global__ __aicore__ void add_custom(GM_ADDR x, GM_ADDR y,
                                                 GM_ADDR z);
#endif

#include <fcntl.h>
#include <sys/stat.h>
#include <unistd.h>

#include <cassert>
#include <cstdio>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>
#define CHECK_ACL(x)                                                    \
  do {                                                                  \
    aclError __ret = x;                                                 \
    if (__ret != ACL_ERROR_NONE) {                                      \
      std::cerr << __FILE__ << ":" << __LINE__ << " aclError:" << __ret \
                << std::endl;                                           \
    }                                                                   \
  } while (0);

int32_t main(int32_t argc, char *argv[]) {
  uint32_t blockDim = 8;
  size_t inputByteSize = 8 * 2048 * sizeof(uint16_t);
  size_t outputByteSize = 8 * 2048 * sizeof(uint16_t);

  CHECK_ACL(aclInit(nullptr));
  int32_t deviceId = 0;
  CHECK_ACL(aclrtSetDevice(deviceId));
  aclrtStream stream = nullptr;
  CHECK_ACL(aclrtCreateStream(&stream));

  uint8_t *xHost, *yHost, *zHost;
  uint8_t *xDevice, *yDevice, *zDevice;

  CHECK_ACL(aclrtMallocHost((void **)(&xHost), inputByteSize));
  CHECK_ACL(aclrtMallocHost((void **)(&yHost), inputByteSize));
  CHECK_ACL(aclrtMallocHost((void **)(&zHost), outputByteSize));
  CHECK_ACL(
      aclrtMalloc((void **)&xDevice, inputByteSize, ACL_MEM_MALLOC_HUGE_FIRST));
  CHECK_ACL(
      aclrtMalloc((void **)&yDevice, inputByteSize, ACL_MEM_MALLOC_HUGE_FIRST));
  CHECK_ACL(aclrtMalloc((void **)&zDevice, outputByteSize,
                        ACL_MEM_MALLOC_HUGE_FIRST));

  CHECK_ACL(aclrtMemcpy(xDevice, inputByteSize, xHost, inputByteSize,
                        ACL_MEMCPY_HOST_TO_DEVICE));
  CHECK_ACL(aclrtMemcpy(yDevice, inputByteSize, yHost, inputByteSize,
                        ACL_MEMCPY_HOST_TO_DEVICE));

  // add_custom_do(blockDim, stream, xDevice, yDevice, zDevice);
  ACLRT_LAUNCH_KERNEL(add_custom)
  (blockDim, stream, xDevice, yDevice, zDevice);
  CHECK_ACL(aclrtSynchronizeStream(stream));

  CHECK_ACL(aclrtMemcpy(zHost, outputByteSize, zDevice, outputByteSize,
                        ACL_MEMCPY_DEVICE_TO_HOST));

  std::cout << "exec acl_op" << std::endl;
  CHECK_ACL(aclrtFree(xDevice));
  CHECK_ACL(aclrtFree(yDevice));
  CHECK_ACL(aclrtFree(zDevice));
  CHECK_ACL(aclrtFreeHost(xHost));
  CHECK_ACL(aclrtFreeHost(yHost));
  CHECK_ACL(aclrtFreeHost(zHost));

  CHECK_ACL(aclrtDestroyStream(stream));
  CHECK_ACL(aclrtResetDevice(deviceId));
  CHECK_ACL(aclFinalize());
  return 0;
}