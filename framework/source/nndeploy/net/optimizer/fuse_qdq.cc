#include "nndeploy/net/optimizer/fuse_qdq.h"

#include "nndeploy/net/net.h"
namespace nndeploy {
namespace net {

FuseQdq::FuseQdq() : OptPass("FuseQdq") {};
FuseQdq::~FuseQdq() {};

bool FuseQdq::CheckFuseCondition(
    OpWrapper* dequant_op, OpWrapper* conv_op, OpWrapper* quant_op,
    std::vector<TensorWrapper*>& tensor_repository) {
  // 检查 DequantizeLinear 的输入是否为标量 scale
  device::Tensor* dequant_scale = dequant_op->op_->getInput(1);
  if (dequant_scale->getShape()[0] != 1) {
    return false;
  }

  // 检查 QuantizeLinear 的输入是否为标量 scale
  device::Tensor* quant_scale = quant_op->op_->getInput(1);
  if (quant_scale->getShape()[0] != 1) {
    return false;
  }

  return true;
}

bool FuseQdq::IsShapeMatch(const std::vector<int>& shape1,
                           const std::vector<int>& shape2) {
  if (shape1.size() != shape2.size()) {
    return false;
  }
  for (size_t i = 0; i < shape1.size(); ++i) {
    if (shape1[i] != shape2[i]) {
      return false;
    }
  }
  return true;
}

bool FuseQdq::isAllInputConstant(
    const OpWrapper* op_wrapper,
    const std::vector<TensorWrapper*>& tensor_repository) {
  for (auto tensor_wrapper : tensor_repository) {
    auto it = std::find(tensor_wrapper->consumers_.begin(),
                        tensor_wrapper->consumers_.end(), op_wrapper);
    if (it != tensor_wrapper->consumers_.end()) {
      // tensor是Op的输入，且不是常量，则返回false
      if (!tensor_wrapper->is_weight_) {
        return false;
      }
    }
  }
  return true;
}

/**
 *参考 https://onnx.ai/onnx/operators/onnx__QLinearConv.html#qlinearconv-10
 */
base::Status FuseQdq::optimize(std::vector<TensorWrapper*>& tensor_repository,
                               std::vector<OpWrapper*>& op_repository,
                               int begin_op_index) {
  base::Status status = base::kStatusCodeOk;

  // 1. 模式匹配
  std::vector<ir::OpType> types = {ir::kOpTypeDequantizeLinear, ir::kOpTypeConv,
                                   ir::kOpTypeQuantizeLinear};
  begin_op_index =
      seqPatternMatch(tensor_repository, op_repository, types, begin_op_index);
  if (begin_op_index == -1) {
    // 没有找到匹配的模式，直接返回
    return base::kStatusCodeOk;
  }

  bool fuse_success = true;

  // 获取匹配到的算子
  OpWrapper* dequant_op = op_repository[begin_op_index];
  // TODO:从input、weight、bias都可能匹配成功，只有在dequant是针对input的，才继续向下
  // 此时匹配的是weight或者bias的Dq，跳过
  if (isAllInputConstant(dequant_op, tensor_repository)) {
    fuse_success = false;
  } else {
    OpWrapper* conv_op = dequant_op->successors_[0];
    OpWrapper* quant_op = conv_op->successors_[0];

    // 检查是否满足融合条件
    if (!CheckFuseCondition(dequant_op, conv_op, quant_op, tensor_repository)) {
      fuse_success = false;
    }

    // 3. 更新 Conv Op 为 QLinearConv
    // 创建新的 QLinearConv 算子
    // QLinearConv的input顺序：
    // x、x_scale、x_zero_point、w、x_scale、x_zero_point、
    // x_scale、x_zero_point、 Bias（可选）
    op::Op* qlinear_conv_op =
        op::createOp(conv_op->op_->getDeviceType(), conv_op->op_->getName(),
                     ir::kOpTypeQLinearConv, {}, {});

    ir::QLinearConvParam* qlinear_conv_param = new ir::QLinearConvParam();
    qlinear_conv_op->setParam(std::static_pointer_cast<nndeploy::base::Param>(
        std::shared_ptr<nndeploy::ir::QLinearConvParam>(qlinear_conv_param)));

    // 设置 QLinearConv 的输入
    // 输入数据
    qlinear_conv_op->setInput(dequant_op->op_->getInput(0), 0);  // 输入数据
    qlinear_conv_op->setInput(dequant_op->op_->getInput(1), 1);  // 输入 scale
    qlinear_conv_op->setInput(dequant_op->op_->getInput(2),
                              2);  // 输入 zero_point

    // 权重和偏置
    // 找到 Conv 的权重和偏置的原始 DequantizeLinear 算子
    OpWrapper* weight_dequant_op = nullptr;
    OpWrapper* bias_dequant_op = nullptr;
    for (auto& predecessor : conv_op->predecessors_) {
      if (predecessor->op_->getOpType() == ir::kOpTypeDequantizeLinear) {
        // 根据形状来判断，哪个是weight的DequantizeLinear ，哪个是bias的
        if (IsShapeMatch(predecessor->op_->getInput(0)->getShape(),
                         conv_op->op_->getInput(1)->getShape())) {
          weight_dequant_op = predecessor;
        } else if (conv_op->op_->getInput(2) != nullptr &&
                   IsShapeMatch(predecessor->op_->getInput(0)->getShape(),
                                conv_op->op_->getInput(2)->getShape())) {
          bias_dequant_op = predecessor;
        }
      }
    }

    // weight不是Dq的
    if (!weight_dequant_op) {
      fuse_success = false;
    }

    qlinear_conv_op->setInput(weight_dequant_op->op_->getInput(0), 3);  // 权重
    qlinear_conv_op->setInput(weight_dequant_op->op_->getInput(1),
                              4);  // 权重 scale
    qlinear_conv_op->setInput(weight_dequant_op->op_->getInput(2),
                              5);  // 权重 zero_point

    // 输出
    qlinear_conv_op->setInput(quant_op->op_->getInput(1), 6);  // 输出 scale
    qlinear_conv_op->setInput(quant_op->op_->getInput(2),
                              7);  // 输出 zero_point
    qlinear_conv_op->setAllOutput(quant_op->op_->getAllOutput());

    if (bias_dequant_op) {
      qlinear_conv_op->setInput(bias_dequant_op->op_->getInput(0), 8);  // 偏置
    }

    // 设置 QLinearConv 的参数
    ir::ConvParam* conv_param =
        (ir::ConvParam*)(conv_op->op_->getParam().get());
    qlinear_conv_param->auto_pad_ = conv_param->auto_pad_;
    qlinear_conv_param->strides_ = conv_param->strides_;
    qlinear_conv_param->pads_ = conv_param->pads_;
    qlinear_conv_param->dilations_ = conv_param->dilations_;
    qlinear_conv_param->group_ = conv_param->group_;
    qlinear_conv_param->kernel_shape_ = conv_param->kernel_shape_;

    // 2. 更新 tensor_repository
    // 删除中间的 tensor
    // rmOutputTensorAndMaybeDelete(dequant_op, tensor_repository);
    // rmOutputTensorAndMaybeDelete(conv_op, tensor_repository);
    rmOutputTensorAndMaybeDelete(weight_dequant_op, tensor_repository);
    if (bias_dequant_op) {
      rmOutputTensorAndMaybeDelete(bias_dequant_op, tensor_repository);
    }

    // 4. 更新 op_repository
    // 直接替换dq
    if (dequant_op->op_ != nullptr) {
      delete dequant_op->op_;
    }
    dequant_op->op_ = qlinear_conv_op;

    status = seqPatternMatchUpateTensorRepository(
        tensor_repository, op_repository, types, begin_op_index);
    if (status != base::kStatusCodeOk) {
      return status;
    }

    // 删除conv、QuantizeLinear
    status = seqPatternMatchUpateOpRepository(tensor_repository, op_repository,
                                              types, begin_op_index);
    if (status != base::kStatusCodeOk) {
      return status;
    }
  }
  return this->optimize(tensor_repository, op_repository, begin_op_index);
}

TypeOptPassRegister<TypeOptPassCreator<FuseQdq>> g_fuse_qdq_register(
    base::kDeviceTypeCodeCpu, kOptPassTypeFuseQdq, /*优化等级 */ 2);

}  // namespace net
}  // namespace nndeploy