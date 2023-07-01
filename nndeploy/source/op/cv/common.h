
#ifndef _NNDEPLOY_SOURCE_CV_COMMON_H_
#define _NNDEPLOY_SOURCE_CV_COMMON_H_

#include "nndeploy/source/base/basic.h"
#include "nndeploy/source/base/macro.h"
#include "nndeploy/source/base/status.h"
#include "nndeploy/source/cv/common.h"
#include "nndeploy/source/device/buffer.h"
#include "nndeploy/source/device/mat.h"

#ifdef NNDEPLOY_ENABLE_DEVICE_ARM
#define CV_ARM_INCLUDE(x) x
#define CV_ARM_RUN(x)    \
  {                      \
    if (arm_check_##x) { \
      return arm_##x;    \
    }                    \
  }
#else
#define CV_ARM_INCLUDE(x)
#define CV_ARM_RUN(x)
#endif

#ifdef NNDEPLOY_ENABLE_DEVICE_X86
#define CV_X86_INCLUDE(x) x
#define CV_X86_RUN(x)    \
  {                      \
    if (x86_check_##x) { \
      return x86_##x;    \
    }                    \
  }
#else
#define CV_X86_INCLUDE(x)
#define CV_X86_RUN(x)
#endif

#ifdef NNDEPLOY_ENABLE_DEVICE_OPENCL
#define CV_OPENCL_INCLUDE(x) x
#define CV_OPENCL_RUN(x)    \
  {                         \
    if (opencl_check_##x) { \
      return opencl_##x;    \
    }                       \
  }
#else
#define CV_OPENCL_INCLUDE(x)
#define CV_OPENCL_RUN(x)
#endif

#ifdef NNDEPLOY_ENABLE_DEVICE_CUDA
#define CV_CUDA_INCLUDE(x) x
#define CV_CUDA_RUN(x)    \
  {                       \
    if (cuda_check_##x) { \
      return cuda_##x;    \
    }                     \
  }
#else
#define CV_CUDA_INCLUDE(x)
#define CV_CUDA_RUN(x)
#endif

using CvFuncV0 = std::function<nndeploy::base::Status(const device::Mat &src,
                                                      device::Mat &dst, ...)>;

#endif