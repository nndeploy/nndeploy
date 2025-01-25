#ifndef _NNDEPLOY_OP_ASCEND_CL_ASCEND_C_OP_ADD_KERNEL_H_
#define _NNDEPLOY_OP_ASCEND_CL_ASCEND_C_OP_ADD_KERNEL_H_

namespace nndeploy {
namespace op {

struct AddCustomTilingData {
  uint32_t totalLength;
  uint32_t tileNum;
};

}  // namespace op
}  // namespace nndeploy

#endif