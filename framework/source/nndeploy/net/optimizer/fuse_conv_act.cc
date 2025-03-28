
#include "nndeploy/net/optimizer/fuse_conv_act.h"

namespace nndeploy {
namespace net {

FuseConvAct::FuseConvAct() : OptPass("FuseConvAct") {}

FuseConvAct::~FuseConvAct() {}

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
 * 2. 更新tensor_repository
 *    a. 更新Act的输出：其生产者改为Conv
 *    b. 删除Conv的输出
 * 3. 更新Conv Op
 *    a. outputs_改为Act的outputs_；修改op_desc_：输出和参数融合
 * 4. 更新op_repository：
 *    a. 更新Conv：OpWrapper的successors_改为Act的successors_；
 *    b. 更新Act的successors_节点：该节点的前驱节点改为Conv
 *    c. 删除Act
 */
base::Status FuseConvAct::optimize(
    std::vector<TensorWrapper*>& tensor_repository,
    std::vector<OpWrapper*>& op_repository, int begin_op_index) {
  base::Status status = base::kStatusCodeOk;
  // 1. 模式匹配
  std::vector<ir::OpType> matched_types;
  begin_op_index = seqPatternMatch(tensor_repository, op_repository,
                                   this->types, matched_types, begin_op_index);
  if (begin_op_index == -1) {
    return base::kStatusCodeOk;  // 没有找到匹配的模式，直接返回
  }

  OpWrapper* first_op = op_repository[begin_op_index];
  OpWrapper* last_op = first_op->successors_[0];

  std::vector<device::Tensor*> outputs_tensors = last_op->op_->getAllOutput();
  first_op->op_->setAllOutput(outputs_tensors);  // 修改Conv的输出为Act的输出
  // # 修改op_desc_：输出和参数融合
  ir::ConvParam* ConvParam = (ir::ConvParam*)first_op->op_->getParam().get();
  ConvParam->activate_op_ = matched_types.back();
  ConvParam->fused_op_param_ = last_op->op_->getParam();

  // 更新 tensor_repository
  OpWrapper* current_op = first_op;
  // 打断conv的后接tensor
  rmOutputTensorAndMaybeDelete(current_op, tensor_repository);
  // 接到act的op接到input这里
  for (auto& tensor : tensor_repository) {
    if (tensor->producers_.size() != 1) {
      continue;
    }
    if (tensor->producers_[0] == last_op) {
      tensor->producers_[0] = first_op;
    }
  }

  // 4. 更新op_repository
  status = seqPatternMatchUpateOpRepository(tensor_repository, op_repository,
                                            matched_types, begin_op_index);
  if (status != base::kStatusCodeOk) {
    return status;
  }

  return this->optimize(tensor_repository, op_repository, begin_op_index);
}

TypeOptPassRegister<TypeOptPassCreator<FuseConvAct>> g_fuse_conv_act_register(
    base::kDeviceTypeCodeCpu, kOptPassTypeFuseConvAct, /*优化等级 */ 2);

}  // namespace net
}  // namespace nndeploy