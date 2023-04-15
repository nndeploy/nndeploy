
#ifndef _NNDEPLOY_SOURCE_DEVICE_CUDA_CUDA_UTIL_H_
#define _NNDEPLOY_SOURCE_DEVICE_CUDA_CUDA_UTIL_H_

#include "nndeploy/source/base/basic.h"
#include "nndeploy/source/base/glic_stl_include.h"
#include "nndeploy/source/base/log.h"
#include "nndeploy/source/base/macro.h"
#include "nndeploy/source/base/object.h"
#include "nndeploy/source/base/status.h"
#include "nndeploy/source/device/cuda/cuda_include.h"

namespace nndeploy {
namespace device {

#define NNDEPLOY_CUDA_FETAL_ERROR(err)                             \
  {                                                                \
    std::stringstream _where, _message;                            \
    _where << __FILE__ << ':' << __LINE__;                         \
    _message << std::string(err) + "\n"                            \
             << __FILE__ << ':' << __LINE__ << "\nAborting... \n"; \
    NNDEPLOY_LOGE("%s", _message.str().c_str());                   \
    exit(EXIT_FAILURE);                                            \
  }

#define NNDEPLOY_CUDA_CHECK(status)                                 \
  {                                                                 \
    std::stringstream _error;                                       \
    if (cudaSuccess != status) {                                    \
      _error << "Cuda failure: " << cudaGetErrorName(status) << " " \
             << cudaGetErrorString(status);                         \
      NNDEPLOY_CUDA_FETAL_ERROR(_error.str());                      \
    }                                                               \
  }

#define NNDEPLOY_CUDNN_CHECK(status)                              \
  {                                                               \
    std::stringstream _error;                                     \
    if (status != CUDNN_STATUS_SUCCESS) {                         \
      _error << "CUDNN failure: " << cudnnGetErrorString(status); \
      NNDEPLOY_CUDA_FETAL_ERROR(_error.str());                    \
    }                                                             \
  }

#define NNDEPLOY_CUBLAS_CHECK(status)          \
  {                                            \
    std::stringstream _error;                  \
    if (status != CUBLAS_STATUS_SUCCESS) {     \
      _error << "Cublas failure: "             \
             << " " << status;                 \
      NNDEPLOY_CUDA_FETAL_ERROR(_error.str()); \
    }                                          \
  }

/**
 * @brief queries the number of available devices
 */
inline int cudaGetNumDevices() {
  int n = 0;
  NNDEPLOY_CHECK_CUDA(cudaGetDeviceCount(&n));
  return n;
}

/**
 * @brief gets the current device associated with the caller thread
 */
inline int cudaGetDeviceId() {
  int id;
  NNDEPLOY_CHECK_CUDA(cudaGetDevice(&id));
  return id;
}

/**
 * @brief switches to a given device context
 */
inline void cudaSetDeviceId(int id) { NNDEPLOY_CHECK_CUDA(cudaSetDevice(id)); }

/**
 * @brief obtains the device property
 */
inline void cudaGetDeviceProperty(int i, cudaDeviceProp& p) {
  NNDEPLOY_CHECK_CUDA(cudaGetDeviceProperties(&p, i));
}

/**
 * @brief obtains the device property
 */
inline cudaDeviceProp cudaGetDeviceProperty(int i) {
  cudaDeviceProp p;
  NNDEPLOY_CHECK_CUDA(cudaGetDeviceProperties(&p, i));
  return p;
}

/**
 * @brief dumps the device property
 */
inline void cudaDumpDeviceProperty(std::ostream& os, const cudaDeviceProp& p) {
  os << "Major revision number:         " << p.major << '\n'
     << "Minor revision number:         " << p.minor << '\n'
     << "Name:                          " << p.name << '\n'
     << "Total global memory:           " << p.totalGlobalMem << '\n'
     << "Total shared memory per block: " << p.sharedMemPerBlock << '\n'
     << "Total registers per block:     " << p.regsPerBlock << '\n'
     << "Warp size:                     " << p.warpSize << '\n'
     << "Maximum memory pitch:          " << p.memPitch << '\n'
     << "Maximum threads per block:     " << p.maxThreadsPerBlock << '\n';

  os << "Maximum dimension of block:    ";
  for (int i = 0; i < 3; ++i) {
    if (i) os << 'x';
    os << p.maxThreadsDim[i];
  }
  os << '\n';

  os << "Maximum dimenstion of grid:    ";
  for (int i = 0; i < 3; ++i) {
    if (i) os << 'x';
    os << p.maxGridSize[i];
    ;
  }
  os << '\n';

  os << "Clock rate:                    " << p.clockRate << '\n'
     << "Total constant memory:         " << p.totalConstMem << '\n'
     << "Texture alignment:             " << p.textureAlignment << '\n'
     << "Concurrent copy and execution: " << p.deviceOverlap << '\n'
     << "Number of multiprocessors:     " << p.multiProcessorCount << '\n'
     << "Kernel execution timeout:      " << p.kernelExecTimeoutEnabled << '\n'
     << "GPU sharing Host Memory:       " << p.integrated << '\n'
     << "Host page-locked mem mapping:  " << p.canMapHostMemory << '\n'
     << "Alignment for Surfaces:        " << p.surfaceAlignment << '\n'
     << "Device has ECC support:        " << p.ECCEnabled << '\n'
     << "Unified Addressing (UVA):      " << p.unifiedAddressing << '\n';
}

/**
 * @brief queries the maximum threads per block on a device
 */
inline int cudaGetDeviceMaxThreadsPerBlock(int d) {
  int threads = 0;
  NNDEPLOY_CHECK_CUDA(
      cudaDeviceGetAttribute(&threads, cudaDevAttrMaxThreadsPerBlock, d));
  return threads;
}

/**
 * @brief queries the maximum x-dimension per block on a device
 */
inline int cudaGetDeviceMaxXDimPerBlock(int d) {
  int dim = 0;
  NNDEPLOY_CHECK_CUDA(cudaDeviceGetAttribute(&dim, cudaDevAttrMaxBlockDimX, d));
  return dim;
}

/**
 * @brief queries the maximum y-dimension per block on a device
 */
inline int cudaGetDeviceMaxYDimPerBlock(int d) {
  int dim = 0;
  NNDEPLOY_CHECK_CUDA(cudaDeviceGetAttribute(&dim, cudaDevAttrMaxBlockDimY, d));
  return dim;
}

/**
 * @brief queries the maximum z-dimension per block on a device
 */
inline size_t cudaGetDeviceMaxZDimPerBlock(int d) {
  int dim = 0;
  NNDEPLOY_CHECK_CUDA(cudaDeviceGetAttribute(&dim, cudaDevAttrMaxBlockDimZ, d));
  return dim;
}

/**
 * @brief queries the maximum x-dimension per grid on a device
 */
inline size_t cudaGetDeviceMaxXDimPerGrid(int d) {
  int dim = 0;
  NNDEPLOY_CHECK_CUDA(
      cudaDeviceGetAttribute(&dim, cudaDevAttrMaxGridDimX, d),
      "failed to query the maximum x-dimension per grid on device ", d);
  return dim;
}

/**
 * @brief queries the maximum y-dimension per grid on a device
 */
inline size_t cudaGetDeviceMaxYDimPerGrid(int d) {
  int dim = 0;
  NNDEPLOY_CHECK_CUDA(cudaDeviceGetAttribute(&dim, cudaDevAttrMaxGridDimY, d));
  return dim;
}

/**
 * @brief queries the maximum z-dimension per grid on a device
 */
inline size_t cudaGetDeviceMaxZDimPerGrid(int d) {
  int dim = 0;
  NNDEPLOY_CHECK_CUDA(cudaDeviceGetAttribute(&dim, cudaDevAttrMaxGridDimZ, d));
  return dim;
}

/**
 * @brief queries the maximum shared memory size in bytes per block on a
 * device
 */
inline size_t cudaGetDeviceMaxSharedMemoryPerBlock(int d) {
  int num = 0;
  NNDEPLOY_CHECK_CUDA(
      cudaDeviceGetAttribute(&num, cudaDevAttrMaxSharedMemoryPerBlock, d));
  return num;
}

/**
 * @brief queries the warp size on a device
 */
inline size_t cudaGetDeviceWarpSize(int d) {
  int num = 0;
  NNDEPLOY_CHECK_CUDA(cudaDeviceGetAttribute(&num, cudaDevAttrWarpSize, d));
  return num;
}

/**
@brief queries the major number of compute capability of a device
*/
inline int cudaGetDeviceComputeCapabilityMajor(int d) {
  int num = 0;
  NNDEPLOY_CHECK_CUDA(
      cudaDeviceGetAttribute(&num, cudaDevAttrComputeCapabilityMajor, d));
  return num;
}

/**
 * @brief queries the minor number of compute capability of a device
 */
inline int cudaGetDeviceComputeCapabilityMinor(int d) {
  int num = 0;
  NNDEPLOY_CHECK_CUDA(
      cudaDeviceGetAttribute(&num, cudaDevAttrComputeCapabilityMinor, d));
  return num;
}

/**
 * @brief queries if the device supports unified addressing
 */
inline bool cudaGetDeviceUnifiedAddressing(int d) {
  int num = 0;
  NNDEPLOY_CHECK_CUDA(
      cudaDeviceGetAttribute(&num, cudaDevAttrUnifiedAddressing, d));
  return num;
}

/**
 * @brief queries the latest CUDA version (1000 * major + 10 * minor)
 * supported by the driver
 */
inline int cudaGetDriverVersion() {
  int num = 0;
  NNDEPLOY_CHECK_CUDA(cudaDriverGetVersion(&num));
  return num;
}

/**
 * @brief queries the CUDA Runtime version (1000 * major + 10 * minor)
 */
inline int cudaGetRuntimeVersion() {
  int num = 0;
  NNDEPLOY_CHECK_CUDA(cudaRuntimeGetVersion(&num));
  return num;
}

}  // namespace device
}  // namespace nndeploy

#endif /* _NNDEPLOY_SOURCE_DEVICE_CUDA_CUDA_UTIL_H_ */
