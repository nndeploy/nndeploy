
#ifndef _NNDEPLOY_MODEL_PREPROCESS_UTIL_H_
#define _NNDEPLOY_MODEL_PREPROCESS_UTIL_H_

#include "nndeploy/base/common.h"
#include "nndeploy/base/glic_stl_include.h"
#include "nndeploy/base/log.h"
#include "nndeploy/base/macro.h"
#include "nndeploy/base/object.h"
#include "nndeploy/base/status.h"
#include "nndeploy/base/string.h"
#include "nndeploy/base/value.h"
#include "nndeploy/dag/packet.h"
#include "nndeploy/dag/task.h"
#include "nndeploy/device/buffer.h"
#include "nndeploy/device/buffer_pool.h"
#include "nndeploy/device/device.h"
#include "nndeploy/device/tensor.h"
#include "nndeploy/model/preprocess/params.h"

namespace nndeploy {
namespace model {

extern NNDEPLOY_CC_API int getChannelByPixelType(base::PixelType pixel_type);

template <typename T>
void normalizeC1(const T* __restrict src, float* __restrict dst, size_t size,
                 const float* __restrict scale, const float* __restrict mean,
                 const float* __restrict std) {
  const float mul_scale = scale[0] / std[0];
  const float add_bias = -mean[0] / std[0];
  for (size_t i = 0; i < size; ++i) {
    dst[i] = src[i] * mul_scale + add_bias;
  }
}
template <typename T>
void normalizeC2(const T* __restrict src, float* __restrict dst, size_t size,
                 const float* __restrict scale, const float* __restrict mean,
                 const float* __restrict std) {
  const float mul_scale[2] = {scale[0] / std[0], scale[1] / std[1]};
  const float add_bias[2] = {-mean[0] / std[0], -mean[1] / std[1]};
  for (size_t i = 0; i < size * 2; i += 2) {
    dst[i] = src[i] * mul_scale[0] + add_bias[0];
    dst[i + 1] = src[i + 1] * mul_scale[1] + add_bias[1];
  }
}
template <typename T>
void normalizeC3(const T* __restrict src, float* __restrict dst, size_t size,
                 const float* __restrict scale, const float* __restrict mean,
                 const float* __restrict std) {
  const float mul_scale[3] = {scale[0] / std[0], scale[1] / std[1],
                              scale[2] / std[2]};
  const float add_bias[3] = {-mean[0] / std[0], -mean[1] / std[1],
                             -mean[2] / std[2]};
  for (size_t i = 0; i < size * 3; i += 3) {
    dst[i] = src[i] * mul_scale[0] + add_bias[0];
    dst[i + 1] = src[i + 1] * mul_scale[1] + add_bias[1];
    dst[i + 2] = src[i + 2] * mul_scale[2] + add_bias[2];
  }
}
template <typename T>
void normalizeC4(const T* __restrict src, float* __restrict dst, size_t size,
                 const float* __restrict scale, const float* __restrict mean,
                 const float* __restrict std) {
  const float mul_scale[4] = {scale[0] / std[0], scale[1] / std[1],
                              scale[2] / std[2], scale[3] / std[3]};
  const float add_bias[4] = {-mean[0] / std[0], -mean[1] / std[1],
                             -mean[2] / std[2], -mean[3] / std[3]};
  for (size_t i = 0; i < size * 3; i += 3) {
    dst[i] = src[i] * mul_scale[0] + add_bias[0];
    dst[i + 1] = src[i + 1] * mul_scale[1] + add_bias[1];
    dst[i + 2] = src[i + 2] * mul_scale[2] + add_bias[2];
    dst[i + 3] = src[i + 3] * mul_scale[3] + add_bias[3];
  }
}

}  // namespace model
}  // namespace nndeploy

#endif /* _NNDEPLOY_MODEL_PREPROCESS_UTIL_H_ */
