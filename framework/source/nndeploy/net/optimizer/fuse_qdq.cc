#include "nndeploy/net/optimizer/fuse_qdq.h"

#include "nndeploy/net/net.h"
#include "nndeploy/net/util.h"
namespace nndeploy {
namespace net {

FuseQdq::FuseQdq() : OptPass("FuseQdq") {};
FuseQdq::~FuseQdq() {};

int FuseQdq::seqPatternMatch(std::vector<TensorWrapper*>& tensor_repository,
                             std::vector<OpWrapper*>& op_repository,
                             const std::vector<ir::OpType>& types,
                             int begin_op_index) {
  for (int i = begin_op_index; i < op_repository.size(); ++i) {
    OpWrapper* current_op = op_repository[i];
    if (current_op->op_->getOpType() == types[0] &&
        current_op->successors_.size() == 1) {
      bool match = true;
      OpWrapper* next_op = current_op;

      for (int j = 1; j < types.size(); ++j) {
        next_op = next_op->successors_[0];

        if (next_op->op_->getOpType() != types[j]) {
          match = false;
          break;
        }
      }

      // 除最后一个节点外，中间节点的输出tensor不能为模型的输出节点
      OpWrapper* middle_op = current_op;
      for (int k = 0; k < types.size() - 1; ++k) {
        for (TensorWrapper* tensor : tensor_repository) {
          if (tensor->producers_.size() == 1 &&
              tensor->producers_[0] == middle_op &&
              tensor->input_output_type_ == kOutput) {
            match = false;
            break;
          }
        }
        middle_op = middle_op->successors_[0];
      }

      // 如果匹配，则返回当前op的index
      if (match) {
        return i;
      }
    }
  }

  return -1;
}

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

  if (begin_op_index >= op_repository.size()) {
    return status;
  }

  // 模式匹配
  std::vector<ir::OpType> types = {ir::kOpTypeDequantizeLinear, ir::kOpTypeConv,
                                   ir::kOpTypeQuantizeLinear};
  begin_op_index =
      seqPatternMatch(tensor_repository, op_repository, types, begin_op_index);
  if (begin_op_index == -1) {
    // 没有找到匹配的模式，直接返回
    return base::kStatusCodeOk;
  }

  // 获取匹配到的算子
  OpWrapper* dequant_op = op_repository[begin_op_index];
  // 从input、weight、bias都可能匹配成功，只有在dequant是针对input的，才继续向下
  // 此时匹配的是weight或者bias的Dq，跳过
  if (isAllInputConstant(dequant_op, tensor_repository)) {
    return this->optimize(tensor_repository, op_repository, begin_op_index + 1);
  } else {
    OpWrapper* conv_op = dequant_op->successors_[0];
    OpWrapper* quant_op = conv_op->successors_[0];

    // 检查是否满足融合条件
    if (!CheckFuseCondition(dequant_op, conv_op, quant_op, tensor_repository)) {
      return this->optimize(tensor_repository, op_repository,
                            begin_op_index + 1);
    }

    // 更新 Conv Op 为 QLinearConv
    // 创建新的 QLinearConv 算子
    // QLinearConv的input顺序：
    // x、x_scale、x_zero_point、w、x_scale、x_zero_point、
    // x_scale、x_zero_point、 Bias（可选）
    op::Op* qlinear_conv_op =
        op::createOp(conv_op->op_->getDeviceType(),
                     "QLinearConv(FuseQdqPass)" + std::to_string(qdq_index_),
                     ir::kOpTypeQLinearConv, {}, {});

    auto qlinear_conv_param =
        std::make_shared<ir::QLinearConvParam>();   
    qlinear_conv_op->setParam(qlinear_conv_param);  

    // 设置 QLinearConv 的输入
    // 输入数据
    qlinear_conv_op->setInput(dequant_op->op_->getInput(0), 0);  // 输入数据
    qlinear_conv_op->setInput(dequant_op->op_->getInput(1), 1);  // 输入 scale
    qlinear_conv_op->setInput(dequant_op->op_->getInput(2),
                              2);  // 输入 zero_point

    // 找到 Conv 的qweight和bias的原始 DequantizeLinear 算子
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
      return this->optimize(tensor_repository, op_repository,
                            begin_op_index + 1);
    }

    // 从这里开始，可以进行算子融合与替换
    qdq_index_++;

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

    // 删除中间的 tensor
    status = seqPatternMatchUpateTensorRepository(
        tensor_repository, op_repository, types, begin_op_index);

    rmOutputTensorAndMaybeDelete(conv_op, tensor_repository);
    rmOutputTensorAndMaybeDelete(weight_dequant_op, tensor_repository);
    if (bias_dequant_op) {
      rmOutputTensorAndMaybeDelete(bias_dequant_op, tensor_repository);
    }

    for (auto tensor_wrapper : tensor_repository) {
      auto it = std::find(tensor_wrapper->consumers_.begin(),
                          tensor_wrapper->consumers_.end(), quant_op);
      if (it != tensor_wrapper->consumers_.end()) {
        tensor_wrapper->consumers_.erase(it);
      }
    }

    // 删除原本的Conv、QuantizeLinear算子
    status = seqPatternMatchUpateOpRepository(tensor_repository, op_repository,
                                              types, begin_op_index);

    dequant_op->op_ = qlinear_conv_op;
    dequant_op->name_ = "QLinearConv(FuseQdqPass)" + std::to_string(qdq_index_);

    // 删除对weight、bias的DequantizeLinear算子
    std::set<OpWrapper*> to_delete_ops = {weight_dequant_op, bias_dequant_op};

    for (auto op_wrapper : to_delete_ops) {
      for (auto tensor_wrapper : tensor_repository) {
        if (tensor_wrapper->consumers_.size() >= 1 &&
            tensor_wrapper->consumers_[0] == op_wrapper) {
          tensor_wrapper->consumers_[0] = dequant_op;
        }
      }

      auto it =
          std::find(op_repository.begin(), op_repository.end(), op_wrapper);
      if (it != op_repository.end()) {
        if (op_wrapper->op_ != nullptr) {
          delete op_wrapper->op_;
          op_wrapper->op_ = nullptr;
        }
        op_repository.erase(it);
        NNDEPLOY_LOGE("delete op name: %s\n", op_wrapper->name_.c_str());
        delete op_wrapper;
      }
    }

    if (status != base::kStatusCodeOk) {
      return status;
    }

    // 查看op_wrapper、tensor_wrapper的前驱后继关系
    // printNetInfo(op_repository, tensor_repository);

    // 这里继续从0开始，因为中途删过几个Op，导致begin_op_index应该前移；
    // 由于算子出现顺序等原因，不易判断前移几个，因此直接从0开始
    return this->optimize(tensor_repository, op_repository, 0);
  }
}

TypeOptPassRegister<TypeOptPassCreator<FuseQdq>> g_fuse_qdq_register(
    base::kDeviceTypeCodeCpu, kOptPassTypeFuseQdq, /*优化等级 */ 2);

}  // namespace net
}  // namespace nndeploy