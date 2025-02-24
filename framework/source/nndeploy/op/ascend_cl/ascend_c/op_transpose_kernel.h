#ifndef _NNDEPLOY_OP_ASCEND_CL_ASCEND_C_OP_TRANSPOSE_KERNEL_H_
#define _NNDEPLOY_OP_ASCEND_CL_ASCEND_C_OP_TRANSPOSE_KERNEL_H_

namespace nndeploy {
namespace op {

struct TransposeTilingData {
  uint32_t N;
  uint32_t C;
  uint32_t H;
  uint32_t W;
};

}  // namespace op
}  // namespace nndeploy

#endif