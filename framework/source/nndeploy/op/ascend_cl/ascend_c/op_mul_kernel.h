#ifndef _NNDEPLOY_OP_ASCEND_CL_ASCEND_C_OP_MUL_KERNEL_H_
#define _NNDEPLOY_OP_ASCEND_CL_ASCEND_C_OP_MUL_KERNEL_H_

namespace nndeploy {
namespace op {

struct MulCustomTilingData {
  uint32_t totalLength;
  uint32_t tileNum;
};

}  // namespace op
}  // namespace nndeploy

#endif