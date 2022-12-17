/**
 * @file half.h
 * @author your name (you@domain.com)
 * @brief 
 * @version 0.1
 * @date 2022-11-24
 * 
 * @copyright Copyright (c) 2022
 * 
 */

#ifndef _NN_DEPLOY_BASE_HALF_
#define _NN_DEPLOY_BASE_HALF_

#include "nn_deploy/base/include_c_cpp.h"

namespace nn_deploy {
namespace base {

bool ConvertFromFloatToBfp16(float *fp32, void *bfp16, int count);

bool ConvertFromBfp16ToFloat(void *bfp16, float *fp32, int count);

bool ConvertFromFloatToFp16(float *fp32, void *fp16, int count);

bool ConvertFromFp16ToFloat(void *fp16, float *fp32, int count);

}
}

#endif