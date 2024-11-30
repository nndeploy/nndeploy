

#ifndef _NNDEPLOY_NET_OPTIMIZER_FUSE_CONV_ACT_H_
#define _NNDEPLOY_NET_OPTIMIZER_FUSE_CONV_ACT_H_

#include "nndeploy/net/optimizer.h"

namespace nndeploy {
namespace net {

class FuseConvAct : public OptPass {
 public:
  FuseConvAct();
  virtual ~FuseConvAct();

  /*
   * @brief 融合Conv和Act
   * @param tensor_repository
   * @param op_repository
   * @return
   * @note
   * 1. 模式匹配：遍历op_repository，找到满足融合的条件：
   *    a. 找到Conv+Act的组合，
   *    b. Conv的输出仅为Act的输入
   *    c. Act的输入仅为Conv的输出
   * 2. 更新tensor_repository conv->act
   *    a. 更新Act的输出：其生产者改为Conv
   *    b. 删除Conv的输出
   * 3. 更新Conv Op
   *    a. outputs_改为Act的outputs_；
   *    b. 修改op_desc_：输出和参数融合
   * 4. 更新op_repository：
   *    a. 更新Conv：OpWrapper的successors_改为Act的successors_；
   *    b. 更新Act的successors_节点：该节点的前驱节点改为Conv
   *    c. 删除Act
   */
  virtual base::Status optimize(std::vector<TensorWrapper*>& tensor_repository,
                                std::vector<OpWrapper*>& op_repository,
                                int begin_op_index) final;

  int seqPatternMatch(std::vector<TensorWrapper*>& tensor_repository,
                      std::vector<OpWrapper*>& op_repository,
                      const std::vector<OpSet>& types,
                      std::vector<ir::OpType>& matched_types,
                      int begin_op_index);
  std::vector<OpSet> types{{ir::kOpTypeConv},  // first conv_type
                           {ir::kOpTypeRelu, ir::kOpTypeSigmoid}};
};

}  // namespace net
}  // namespace nndeploy

#endif /* _NNDEPLOY_NET_OPTIMIZER_FUSE_CONV_ACT_H_ */
