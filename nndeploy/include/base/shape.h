/**
 * @file shape.h
 * @author your name (you@domain.com)
 * @brief
 * @version 0.1
 * @date 2022-12-20
 *
 * @copyright Copyright (c) 2022
 * @note 参考tnn oneflow
 *
 */
#ifndef _NNDEPLOY_INCLUDE_BASE_SHAPE_H_
#define _NNDEPLOY_INCLUDE_BASE_SHAPE_H_

#include "nndeploy/include/base/include_c_cpp.h"
#include "nndeploy/include/base/macro.h"
#include "nndeploy/include/base/type.h"

namespace nndeploy {
namespace base {

int shapeCount(const IntVector &dims, int start_index = 0, int end_index = -1);

IntVector shapeMax(const IntVector &dims0, const IntVector &dims1,
                   int start_index = 0, int end_index = -1);

IntVector shapeMin(const IntVector &dims0, const IntVector &dims1,
                   int start_index = 0, int end_index = -1);

bool shapeEqual(const IntVector &dims0, const IntVector &dims1,
                int start_index = 0, int end_index = -1);

IntVector shapeNchw2Nhwc(const IntVector &dims);

IntVector shapeNhwc2Nchw(const IntVector &dims);

}  // namespace base
}  // namespace nndeploy

#endif