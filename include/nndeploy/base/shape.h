
#ifndef _NNDEPLOY_BASE_SHAPE_H_
#define _NNDEPLOY_BASE_SHAPE_H_

#include "nndeploy/base/common.h"
#include "nndeploy/base/glic_stl_include.h"
#include "nndeploy/base/macro.h"

namespace nndeploy {
namespace base {

extern NNDEPLOY_CC_API size_t shapeCount(const IntVector &dims,
                                         int start_index = 0,
                                         int end_index = -1);

extern NNDEPLOY_CC_API IntVector shapeMax(const IntVector &dims0,
                                          const IntVector &dims1,
                                          int start_index = 0,
                                          int end_index = -1);

extern NNDEPLOY_CC_API IntVector shapeMin(const IntVector &dims0,
                                          const IntVector &dims1,
                                          int start_index = 0,
                                          int end_index = -1);

extern NNDEPLOY_CC_API bool shapeEqual(const IntVector &dims0,
                                       const IntVector &dims1,
                                       int start_index = 0, int end_index = -1);

extern NNDEPLOY_CC_API IntVector shapeNchw2Nhwc(const IntVector &dims);

extern NNDEPLOY_CC_API IntVector shapeNhwc2Nchw(const IntVector &dims);

}  // namespace base
}  // namespace nndeploy

#endif