

#ifndef _NNDEPLOY_NET_OPTIMIZER_FUSE_CONV_RELU_H_
#define _NNDEPLOY_NET_OPTIMIZER_FUSE_CONV_RELU_H_

#include "nndeploy/net/optimizer.h"

namespace nndeploy {
namespace net {


class FuseConvRelu : public OptPass {
 public:
  FuseConvRelu();
  virtual ~FuseConvRelu();

  /*
   * @brief 融合Conv和Relu
   * @param tensor_repository 
   * @param op_repository 
   * @return
   * @note
   * 1. 模式匹配：遍历op_repository，找到满足融合的条件：
   *    a. 找到Conv+Relu的组合，
   *    b. Conv的输出仅为Relu的输入
   *    c. Relu的输入仅为Conv的输出
   * 2. 更新tensor_repository conv->relu
   *    a. 更新Relu的输出：其生产者改为Conv
   *    b. 删除Conv的输出
   * 3. 更新Conv Op
   *    a. outputs_改为Relu的outputs_；
   *    b. 修改op_desc_：输出和参数融合
   * 4. 更新op_repository：
   *    a. 更新Conv：OpWrapper的successors_改为Relu的successors_；
   *    b. 更新Relu的successors_节点：该节点的前驱节点改为Conv
   *    c. 删除Relu
   */
  virtual base::Status optimize(std::vector<TensorWrapper *>& tensor_repository,
                                std::vector<OpWrapper *>& op_repository, int begin_op_index);
};

}  // namespace net
}  // namespace nndeploy

#endif /* _NNDEPLOY_NET_OPTIMIZER_FUSE_CONV_RELU_H_ */
