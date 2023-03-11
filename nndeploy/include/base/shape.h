
#ifndef _NNDEPLOY_INCLUDE_BASE_SHAPE_H_
#define _NNDEPLOY_INCLUDE_BASE_SHAPE_H_

#include "nndeploy/include/base/basic.h"
#include "nndeploy/include/base/include_c_cpp.h"
#include "nndeploy/include/base/macro.h"

namespace nndeploy {
namespace base {

size_t shapeCount(const IntVector &dims, int start_index = 0,
                  int end_index = -1);

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