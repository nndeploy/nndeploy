
#include "nndeploy/net/optimizer/fuse_conv_relu.h"

namespace nndeploy {
namespace net {

FuseConvRelu::FuseConvRelu() {}

FuseConvRelu::~FuseConvRelu() {}

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
 * 2. 更新tensor_repository
 *    a. 更新Relu的输出：其生产者改为Conv
 *    b. 删除Conv的输出
 * 3. 更新Conv Op
 *    a. outputs_改为Relu的outputs_；修改op_desc_：输出和参数融合
 * 4. 更新op_repository：
 *    a. 更新Conv：OpWrapper的successors_改为Relu的successors_；
 *    b. 更新Relu的successors_节点：该节点的前驱节点改为Conv
 *    c. 删除Relu
 */
base::Status FuseConvRelu::optimize(
    std::vector<TensorWrapper*>& tensor_repository,
    std::vector<OpWrapper*>& op_repository, int begin_op_index) {
  // 1. 模式匹配
  std::vector<ir::OpType> types = {ir::kOpTypeConv, ir::kOpTypeRelu};
  begin_op_index =
      seqPatternMatch(tensor_repository, op_repository, types, begin_op_index);
  if (begin_op_index == -1) {
    return base::kStatusCodeOk;  // 没有找到匹配的模式，直接返回
  }

  // 2. 更新tensor_repository
  base::Status status = seqPatternMatchUpateTensorRepository(
      tensor_repository, op_repository, types, begin_op_index);
  if (status != base::kStatusCodeOk) {
    return status;
  }

  // 3. 更新Conv Op
  OpWrapper* first_op = op_repository[begin_op_index];
  OpWrapper* last_op = first_op->successors_[0];
  std::vector<device::Tensor*> outputs_tensors = last_op->op_->getAllOutput();
  first_op->op_->setAllOutput(outputs_tensors);  // 修改Conv的输出为Relu的输出
  // # 修改op_desc_：输出和参数融合
  ir::ConvParam* ConvParam = (ir::ConvParam*)first_op->op_->getParam().get();
  ConvParam->is_fusion_op_ = true;
  ConvParam->activate_op_ = ir::kOpTypeRelu;

  // 4. 更新op_repository
  status = seqPatternMatchUpateOpRepository(tensor_repository, op_repository,
                                            types, begin_op_index);
  if (status != base::kStatusCodeOk) {
    return status;
  }

  return this->optimize(tensor_repository, op_repository, begin_op_index);
}

TypeOptPassRegister<TypeOptPassCreator<FuseConvRelu>> g_fuse_conv_relu_register(
    base::kDeviceTypeCodeCpu, kOptPassTypeFuseConvRelu);

}  // namespace net
}  // namespace nndeploy