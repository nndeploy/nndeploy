#ifndef _NNDEPLOY_OP_ASCEND_CL_ASCEND_C_OP_CONV_KERNEL_H_
#define _NNDEPLOY_OP_ASCEND_CL_ASCEND_C_OP_CONV_KERNEL_H_

#include <stdint.h>

namespace nndeploy {
namespace op {

struct Conv2dTilingData {
  uint32_t batchSize;
  uint32_t inHeight;
  uint32_t inWidth;
  uint32_t outHeight;
  uint32_t outWidth;
  uint32_t kernelSize;
  uint32_t inChannel;
  uint32_t outChannel;
  uint32_t stride;
  uint32_t padding;
  uint32_t dilation;
  uint32_t coreNum;
};

}  // namespace op
}  // namespace nndeploy

#endif