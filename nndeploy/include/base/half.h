/**
 * @file half.h
 * @author your name (you@domain.com)
 * @brief
 * @version 0.1
 * @date 2022-12-21
 *
 * @copyright Copyright (c) 2022
 *
 */
#ifndef _NNDEPLOY_INCLUDE_BASE_HALF_H_
#define _NNDEPLOY_INCLUDE_BASE_HALF_H_

#include "nndeploy/include/base/include_c_cpp.h"

namespace nndeploy {
namespace base {

bool convertFromFloatToBfp16(float *fp32, void *bfp16, int count);

bool convertFromBfp16ToFloat(void *bfp16, float *fp32, int count);

bool convertFromFloatToFp16(float *fp32, void *fp16, int count);

bool convertFromFp16ToFloat(void *fp16, float *fp32, int count);

}  // namespace base
}  // namespace nndeploy

#endif