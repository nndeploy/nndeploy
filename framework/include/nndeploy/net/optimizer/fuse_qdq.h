#ifndef _NNDEPLOY_NET_OPTIMIZER_FUSE_QDQ_H_
#define _NNDEPLOY_NET_OPTIMIZER_FUSE_QDQ_H_

#include "nndeploy/net/optimizer.h"

namespace nndeploy {

namespace net {

/**
 * QDQ算子融合 包括：
 * 1. dq + conv + q  -> qLinearConv
 * 2. dq + matmul +q -> qLinearMatmul (暂未支持)
 *
 * 当前只支持Per-Tensor的量化方式融合
 */
class FuseQdq : public OptPass {
 public:
  FuseQdq();
  virtual ~FuseQdq();

  virtual base::Status optimize(std::vector<TensorWrapper*>& tensor_repository,
                                std::vector<OpWrapper*>& op_repository,
                                int begin_op_index);

  bool CheckFuseCondition(OpWrapper* dequant_op, OpWrapper* conv_op,
                          OpWrapper* quant_op,
                          std::vector<TensorWrapper*>& tensor_repository);

  bool IsShapeMatch(const std::vector<int>& shape1,
                    const std::vector<int>& shape2);

  bool isAllInputConstant(
      const OpWrapper* op_wrapper,
      const std::vector<TensorWrapper*>& tensor_repository);
};
}  // namespace net
}  // namespace nndeploy

#endif