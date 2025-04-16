#include "nndeploy/net/optimizer/fold_constant.h"

#include "nndeploy/ir/op_param.h"

namespace nndeploy {

namespace net {

FoldConstant::FoldConstant() : OptPass("FoldConstant") {};
FoldConstant::~FoldConstant() {};

/**
 * 是否是确定性算子
 * 无论执行多少次，节点都会产生相同的输出。如果节点涉及任何非确定性操作（例如随机数生成），则不能进行常量折叠。
 */
bool FoldConstant::isDeterministic(const OpWrapper* op_wrapper) {
  std::set<ir::OpType> non_deterministic_ops{
      ir::kOpTypeRandomNormal,
      ir::kOpTypeRandomNormalLike,
      ir::kOpTypeRandomUniform,
      ir::kOpTypeRandomUniformLike,
  };

  auto it =
      std::find(non_deterministic_ops.begin(), non_deterministic_ops.end(),
                op_wrapper->op_->getOpType());
  return it == non_deterministic_ops.end();
}

/**
 * 是否是量化相关算子
 */
bool FoldConstant::isQDQ(const OpWrapper* op_wrapper) {
  std::set<ir::OpType> quant_ops{ir::kOpTypeQuantizeLinear,
                                 ir::kOpTypeDequantizeLinear};
  auto it = std::find(quant_ops.begin(), quant_ops.end(),
                      op_wrapper->op_->getOpType());
  return it != quant_ops.end();
}

/**
 * 输入是否都是常量
 */
bool FoldConstant::isAllInputConstant(
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
 * TODO:参考OnnxSim
 * 输出张量大小不超过特定阈值：如果节点的输出张量过大，可能会超出内存限制或导致性能问题，因此可能需要避免常量折叠。
 * 当前暂时全部返回false
 */
bool FoldConstant::produceLargeTensor(
    const OpWrapper* op_wrapper,
    const std::vector<TensorWrapper*>& tensor_repository) {
  return false;
}

/**
 * 该算子是否支持常量折叠
 * 有一些需要获取运行时信息的算子不支持折叠
 */
bool FoldConstant::isSupportFold(const OpWrapper* op_wrapper) { return true; }
/**
 * 运行这个可折叠的Op
 * 在Net构建时该op的input、output都已set，但是output没有分配内存
 */
base::Status FoldConstant::runOp(const OpWrapper* op_wrapper) {
  base::Status status = base::kStatusCodeOk;
  op::Op* op = op_wrapper->op_;

  status = op->init();
  NNDEPLOY_RETURN_ON_NEQ(status, base::kStatusCodeOk, "init failed");
  status = op->preRun();
  NNDEPLOY_RETURN_ON_NEQ(status, base::kStatusCodeOk, "preRun failed");
  status = op->checkOrAllocOutput();
  NNDEPLOY_RETURN_ON_NEQ(status, base::kStatusCodeOk,
                         "checkOrAllocOutput failed");
  status = op->run();
  NNDEPLOY_RETURN_ON_NEQ(status, base::kStatusCodeOk, "run failed");
  status = op->postRun();
  NNDEPLOY_RETURN_ON_NEQ(status, base::kStatusCodeOk, "postRun failed");
  status = op->deinit();
  NNDEPLOY_RETURN_ON_NEQ(status, base::kStatusCodeOk, "deinit failed");

  return status;
}

base::Status FoldConstant::optimize(
    std::vector<TensorWrapper*>& tensor_repository,
    std::vector<OpWrapper*>& op_repository, int begin_op_index) {
  if (begin_op_index < 0 || begin_op_index >= op_repository.size()) {
    return base::kStatusCodeOk;
  }

  OpWrapper* op_wrapper = op_repository[begin_op_index];
  bool fold_flag = false;

  if (isDeterministic(op_wrapper) && !isQDQ(op_wrapper) &&
      !produceLargeTensor(op_wrapper, tensor_repository) &&
      isAllInputConstant(op_wrapper, tensor_repository) &&
      isSupportFold(op_wrapper)) {
    base::Status status = runOp(op_wrapper);
    if (status == base::kStatusCodeOk) {
      fold_flag = true;

      // 这个算子已经提前计算 ，将其删除， 思路与删除死节点类似

      // 将已经计算好的输出Tensor，标记为常量
      for (auto tensor_wrapper : tensor_repository) {
        auto it = std::find(tensor_wrapper->producers_.begin(),
                            tensor_wrapper->producers_.end(), op_wrapper);
        if (it != tensor_wrapper->producers_.end()) {
          tensor_wrapper->is_weight_ = true;
          tensor_wrapper->producers_ = {};  // 该Tensor被标记为常量，没有生产者
        }
      }

      // 处理这个Op的输入Tensor
      rmInputTensorAndMaybeDelete(op_wrapper, tensor_repository);

      // 将其从前驱节点的后继节点中删除
      rmOpFromPredecessor(op_wrapper);

      // 将其从后继节点的前驱节点中删除
      rmOpFromSuccessors(op_wrapper);

      // 将待删除Op从Op仓库删除
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
  }

  if (fold_flag) {
    // 由于消除了一个Op，所以下次索引不加1
    return optimize(tensor_repository, op_repository, begin_op_index);
  } else {
    return optimize(tensor_repository, op_repository, begin_op_index + 1);
  }
}

TypeOptPassRegister<TypeOptPassCreator<FoldConstant>> g_fold_constant_register(
    base::kDeviceTypeCodeCpu, kOptPassTypeFoldConstant, 5);

}  // namespace net

}  // namespace nndeploy