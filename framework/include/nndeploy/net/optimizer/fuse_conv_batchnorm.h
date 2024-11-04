#ifndef _NNDEPLOY_NET_OPTIMIZER_FUSE_CONV_BATCHNORM_H_
#define _NNDEPLOY_NET_OPTIMIZER_FUSE_CONV_BATCHNORM_H_

#include "nndeploy/net/optimizer.h"

namespace nndeploy {

namespace net

{

// TODO:当前仅支持float类型 和 OIHW格式得的卷积
// 卷积公式抽象为 z = weight * x + conv_bias
// batchnorm公式为 ：y = （ z - mean）/sqrt(var) * gamma + beta
// 将卷积带入，可得
// weight = weight / sqrt(var) * gamma
// conv_bias = (conv_bias - mean) / srqt(var) * gamma + beta
class FuseConvBatchNorm : public OptPass {
 public:
  FuseConvBatchNorm();
  virtual ~FuseConvBatchNorm();

  /**
   * @brief 将BatchNorm的权重融合进Conv中
   */
  virtual base::Status optimize(std::vector<TensorWrapper*>& tensor_repository,
                                std::vector<OpWrapper*>& op_repository,
                                int begin_op_index);
};

}  // namespace net

}  // namespace nndeploy

#endif