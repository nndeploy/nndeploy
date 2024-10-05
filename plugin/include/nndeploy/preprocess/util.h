
#ifndef _NNDEPLOY_PREPROCESS_UTIL_H_
#define _NNDEPLOY_PREPROCESS_UTIL_H_

#include "nndeploy/base/common.h"
#include "nndeploy/base/glic_stl_include.h"
#include "nndeploy/base/half.h"
#include "nndeploy/base/log.h"
#include "nndeploy/base/macro.h"
#include "nndeploy/base/object.h"
#include "nndeploy/base/status.h"
#include "nndeploy/base/string.h"
#include "nndeploy/base/value.h"
#include "nndeploy/dag/edge.h"
#include "nndeploy/dag/node.h"
#include "nndeploy/device/buffer.h"
#include "nndeploy/device/device.h"
#include "nndeploy/device/memory_pool.h"
#include "nndeploy/device/tensor.h"
#include "nndeploy/preprocess/params.h"

namespace nndeploy {
namespace preprocess {

extern NNDEPLOY_CC_API int getChannelByPixelType(base::PixelType pixel_type);

template <typename T>
void normalizeFp16C1(const T *__restrict src, void *dst, size_t size,
                     const float *__restrict scale,
                     const float *__restrict mean,
                     const float *__restrict std) {
  float *dst_tmp = (float *)malloc(size * sizeof(float));
  const float mul_scale = scale[0] / std[0];
  const float add_bias = -mean[0] / std[0];
  for (size_t i = 0; i < size; ++i) {
    dst_tmp[i] = src[i] * mul_scale + add_bias;
  }
  base::convertFromFloatToFp16(dst_tmp, dst, size);
  free(dst_tmp);
}
template <typename T>
void normalizeFp16C2(const T *__restrict src, void *dst, size_t size,
                     const float *__restrict scale,
                     const float *__restrict mean,
                     const float *__restrict std) {
  float *dst_tmp = (float *)malloc(size * 2 * sizeof(float));
  const float mul_scale[2] = {scale[0] / std[0], scale[1] / std[1]};
  const float add_bias[2] = {-mean[0] / std[0], -mean[1] / std[1]};
  for (size_t i = 0; i < size * 2; i += 2) {
    dst_tmp[i] = src[i] * mul_scale[0] + add_bias[0];
    dst_tmp[i + 1] = src[i + 1] * mul_scale[1] + add_bias[1];
  }
  base::convertFromFloatToFp16(dst_tmp, dst, size * 2);
  free(dst_tmp);
}
template <typename T>
void normalizeFp16C3(const T *__restrict src, void *dst, size_t size,
                     const float *__restrict scale,
                     const float *__restrict mean,
                     const float *__restrict std) {
  float *dst_tmp = (float *)malloc(size * 3 * sizeof(float));
  const float mul_scale[3] = {scale[0] / std[0], scale[1] / std[1],
                              scale[2] / std[2]};
  const float add_bias[3] = {-mean[0] / std[0], -mean[1] / std[1],
                             -mean[2] / std[2]};
  for (size_t i = 0; i < size * 3; i += 3) {
    dst_tmp[i] = src[i] * mul_scale[0] + add_bias[0];
    dst_tmp[i + 1] = src[i + 1] * mul_scale[1] + add_bias[1];
    dst_tmp[i + 2] = src[i + 2] * mul_scale[2] + add_bias[2];
  }
  base::convertFromFloatToFp16(dst_tmp, dst, size * 3);
  free(dst_tmp);
}
template <typename T>
void normalizeFp16C4(const T *__restrict src, void *dst, size_t size,
                     const float *__restrict scale,
                     const float *__restrict mean,
                     const float *__restrict std) {
  float *dst_tmp = (float *)malloc(size * 4 * sizeof(float));
  const float mul_scale[4] = {scale[0] / std[0], scale[1] / std[1],
                              scale[2] / std[2], scale[3] / std[3]};
  const float add_bias[4] = {-mean[0] / std[0], -mean[1] / std[1],
                             -mean[2] / std[2], -mean[3] / std[3]};
  for (size_t i = 0; i < size * 3; i += 3) {
    dst_tmp[i] = src[i] * mul_scale[0] + add_bias[0];
    dst_tmp[i + 1] = src[i + 1] * mul_scale[1] + add_bias[1];
    dst_tmp[i + 2] = src[i + 2] * mul_scale[2] + add_bias[2];
    dst_tmp[i + 3] = src[i + 3] * mul_scale[3] + add_bias[3];
  }
  base::convertFromFloatToFp16(dst_tmp, dst, size * 4);
  free(dst_tmp);
}
template <typename T>
void normalizeFp16CN(const T *__restrict src, void *dst, const int c,
                     size_t size, const float *__restrict scale,
                     const float *__restrict mean,
                     const float *__restrict std) {
  float *dst_tmp = (float *)malloc(size * c * sizeof(float));
  float *mul_scale = (float *)malloc(c * sizeof(float));
  float *add_bias = (float *)malloc(c * sizeof(float));
  for (int j = 0; j < c; ++j) {
    mul_scale[j] = scale[j] / std[j];
    add_bias[j] = -mean[j] / std[j];
  }
  for (size_t i = 0; i < size; i++) {
    int ii = i * c;
    for (int j = 0; j < c; ++j) {
      dst_tmp[ii + j] = src[ii + j] * mul_scale[j] + add_bias[j];
    }
  }
  base::convertFromFloatToFp16(dst_tmp, dst, size * c);
  free(dst_tmp);
  free(mul_scale);
  free(add_bias);
}

template <typename T>
void normalizeBfp16C1(const T *__restrict src, void *dst, size_t size,
                      const float *__restrict scale,
                      const float *__restrict mean,
                      const float *__restrict std) {
  float *dst_tmp = (float *)malloc(size * sizeof(float));
  const float mul_scale = scale[0] / std[0];
  const float add_bias = -mean[0] / std[0];
  for (size_t i = 0; i < size; ++i) {
    dst_tmp[i] = src[i] * mul_scale + add_bias;
  }
  base::convertFromFloatToBfp16(dst_tmp, dst, size);
  free(dst_tmp);
}

template <typename T>
void normalizeBfp16C2(const T *__restrict src, void *dst, size_t size,
                      const float *__restrict scale,
                      const float *__restrict mean,
                      const float *__restrict std) {
  float *dst_tmp = (float *)malloc(size * 2 * sizeof(float));
  const float mul_scale[2] = {scale[0] / std[0], scale[1] / std[1]};
  const float add_bias[2] = {-mean[0] / std[0], -mean[1] / std[1]};
  for (size_t i = 0; i < size * 2; i += 2) {
    dst_tmp[i] = src[i] * mul_scale[0] + add_bias[0];
    dst_tmp[i + 1] = src[i + 1] * mul_scale[1] + add_bias[1];
  }
  base::convertFromFloatToBfp16(dst_tmp, dst, size * 2);
  free(dst_tmp);
}
template <typename T>
void normalizeBfp16C3(const T *__restrict src, void *dst, size_t size,
                      const float *__restrict scale,
                      const float *__restrict mean,
                      const float *__restrict std) {
  float *dst_tmp = (float *)malloc(size * 3 * sizeof(float));
  const float mul_scale[3] = {scale[0] / std[0], scale[1] / std[1],
                              scale[2] / std[2]};
  const float add_bias[3] = {-mean[0] / std[0], -mean[1] / std[1],
                             -mean[2] / std[2]};
  for (size_t i = 0; i < size * 3; i += 3) {
    dst_tmp[i] = src[i] * mul_scale[0] + add_bias[0];
    dst_tmp[i + 1] = src[i + 1] * mul_scale[1] + add_bias[1];
    dst_tmp[i + 2] = src[i + 2] * mul_scale[2] + add_bias[2];
  }
  base::convertFromFloatToBfp16(dst_tmp, dst, size * 3);
  free(dst_tmp);
}
template <typename T>
void normalizeBfp16C4(const T *__restrict src, void *dst, size_t size,
                      const float *__restrict scale,
                      const float *__restrict mean,
                      const float *__restrict std) {
  float *dst_tmp = (float *)malloc(size * 4 * sizeof(float));
  const float mul_scale[4] = {scale[0] / std[0], scale[1] / std[1],
                              scale[2] / std[2], scale[3] / std[3]};
  const float add_bias[4] = {-mean[0] / std[0], -mean[1] / std[1],
                             -mean[2] / std[2], -mean[3] / std[3]};
  for (size_t i = 0; i < size * 3; i += 3) {
    dst_tmp[i] = src[i] * mul_scale[0] + add_bias[0];
    dst_tmp[i + 1] = src[i + 1] * mul_scale[1] + add_bias[1];
    dst_tmp[i + 2] = src[i + 2] * mul_scale[2] + add_bias[2];
    dst_tmp[i + 3] = src[i + 3] * mul_scale[3] + add_bias[3];
  }
  base::convertFromFloatToBfp16(dst_tmp, dst, size * 4);
  free(dst_tmp);
}
template <typename T>
void normalizeBfp16CN(const T *__restrict src, void *dst, const int c,
                      size_t size, const float *__restrict scale,
                      const float *__restrict mean,
                      const float *__restrict std) {
  float *dst_tmp = (float *)malloc(size * c * sizeof(float));
  float *mul_scale = (float *)malloc(c * sizeof(float));
  float *add_bias = (float *)malloc(c * sizeof(float));
  for (int j = 0; j < c; ++j) {
    mul_scale[j] = scale[j] / std[j];
    add_bias[j] = -mean[j] / std[j];
  }
  for (size_t i = 0; i < size; i++) {
    int ii = i * c;
    for (int j = 0; j < c; ++j) {
      dst_tmp[ii + j] = src[ii + j] * mul_scale[j] + add_bias[j];
    }
  }
  base::convertFromFloatToBfp16(dst_tmp, dst, size * c);
  free(dst_tmp);
  free(mul_scale);
  free(add_bias);
}

template <typename T>
void normalizeFp32C1(const T *__restrict src, float *__restrict dst,
                     size_t size, const float *__restrict scale,
                     const float *__restrict mean,
                     const float *__restrict std) {
  const float mul_scale = scale[0] / std[0];
  const float add_bias = -mean[0] / std[0];
  for (size_t i = 0; i < size; ++i) {
    dst[i] = src[i] * mul_scale + add_bias;
  }
}
template <typename T>
void normalizeFp32C2(const T *__restrict src, float *__restrict dst,
                     size_t size, const float *__restrict scale,
                     const float *__restrict mean,
                     const float *__restrict std) {
  const float mul_scale[2] = {scale[0] / std[0], scale[1] / std[1]};
  const float add_bias[2] = {-mean[0] / std[0], -mean[1] / std[1]};
  for (size_t i = 0; i < size * 2; i += 2) {
    dst[i] = src[i] * mul_scale[0] + add_bias[0];
    dst[i + 1] = src[i + 1] * mul_scale[1] + add_bias[1];
  }
}
template <typename T>
void normalizeFp32C3(const T *__restrict src, float *__restrict dst,
                     size_t size, const float *__restrict scale,
                     const float *__restrict mean,
                     const float *__restrict std) {
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
void normalizeFp32C4(const T *__restrict src, float *__restrict dst,
                     size_t size, const float *__restrict scale,
                     const float *__restrict mean,
                     const float *__restrict std) {
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
template <typename T>
void normalizeFp32CN(const T *__restrict src, float *__restrict dst,
                     const int c, size_t size, const float *__restrict scale,
                     const float *__restrict mean,
                     const float *__restrict std) {
  float *mul_scale = (float *)malloc(c * sizeof(float));
  float *add_bias = (float *)malloc(c * sizeof(float));
  for (int j = 0; j < c; ++j) {
    mul_scale[j] = scale[j] / std[j];
    add_bias[j] = -mean[j] / std[j];
  }
  for (size_t i = 0; i < size; i++) {
    int ii = i * c;
    for (int j = 0; j < c; ++j) {
      dst[ii + j] = src[ii + j] * mul_scale[j] + add_bias[j];
    }
  }
  free(mul_scale);
  free(add_bias);
}

template <typename T1, typename T2>
void normalizeC1(const T1 *__restrict src, T2 *__restrict dst, size_t size,
                 const float *__restrict scale, const float *__restrict mean,
                 const float *__restrict std) {
  const float mul_scale = scale[0] / std[0];
  const float add_bias = -mean[0] / std[0];
  for (size_t i = 0; i < size; ++i) {
    dst[i] = (T2)(src[i] * mul_scale + add_bias);
  }
}
template <typename T1, typename T2>
void normalizeC2(const T1 *__restrict src, T2 *__restrict dst, size_t size,
                 const float *__restrict scale, const float *__restrict mean,
                 const float *__restrict std) {
  const float mul_scale[2] = {scale[0] / std[0], scale[1] / std[1]};
  const float add_bias[2] = {-mean[0] / std[0], -mean[1] / std[1]};
  for (size_t i = 0; i < size * 2; i += 2) {
    dst[i] = (T2)(src[i] * mul_scale[0] + add_bias[0]);
    dst[i + 1] = (T2)(src[i + 1] * mul_scale[1] + add_bias[1]);
  }
}
template <typename T1, typename T2>
void normalizeC3(const T1 *__restrict src, T2 *__restrict dst, size_t size,
                 const float *__restrict scale, const float *__restrict mean,
                 const float *__restrict std) {
  const float mul_scale[3] = {scale[0] / std[0], scale[1] / std[1],
                              scale[2] / std[2]};
  const float add_bias[3] = {-mean[0] / std[0], -mean[1] / std[1],
                             -mean[2] / std[2]};
  for (size_t i = 0; i < size * 3; i += 3) {
    dst[i] = (T2)(src[i] * mul_scale[0] + add_bias[0]);
    dst[i + 1] = (T2)(src[i + 1] * mul_scale[1] + add_bias[1]);
    dst[i + 2] = (T2)(src[i + 2] * mul_scale[2] + add_bias[2]);
  }
}
template <typename T1, typename T2>
void normalizeC4(const T1 *__restrict src, T2 *__restrict dst, size_t size,
                 const float *__restrict scale, const float *__restrict mean,
                 const float *__restrict std) {
  const float mul_scale[4] = {scale[0] / std[0], scale[1] / std[1],
                              scale[2] / std[2], scale[3] / std[3]};
  const float add_bias[4] = {-mean[0] / std[0], -mean[1] / std[1],
                             -mean[2] / std[2], -mean[3] / std[3]};
  for (size_t i = 0; i < size * 3; i += 3) {
    dst[i] = (T2)(src[i] * mul_scale[0] + add_bias[0]);
    dst[i + 1] = (T2)(src[i + 1] * mul_scale[1] + add_bias[1]);
    dst[i + 2] = (T2)(src[i + 2] * mul_scale[2] + add_bias[2]);
    dst[i + 3] = (T2)(src[i + 3] * mul_scale[3] + add_bias[3]);
  }
}
template <typename T1, typename T2>
void normalizeCN(const T1 *__restrict src, T2 *__restrict dst, const int c,
                 size_t size, const float *__restrict scale,
                 const float *__restrict mean, const float *__restrict std) {
  float *mul_scale = (float *)malloc(c * sizeof(float));
  float *add_bias = (float *)malloc(c * sizeof(float));
  for (int j = 0; j < c; ++j) {
    mul_scale[j] = scale[j] / std[j];
    add_bias[j] = -mean[j] / std[j];
  }
  for (size_t i = 0; i < size; i++) {
    int ii = i * c;
    for (size_t j = 0; j < c; ++j) {
      dst[ii + j] = (T2)(src[ii + j] * mul_scale[j] + add_bias[j]);
    }
  }
  free(mul_scale);
  free(add_bias);
}

}  // namespace preprocess
}  // namespace nndeploy

#endif /* _NNDEPLOY_PREPROCESS_UTIL_H_ */
