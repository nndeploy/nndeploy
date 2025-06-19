#include "nndeploy/net/optimizer/eliminate_dead_op.h"

namespace nndeploy {

namespace net {
EliminateDeadOp::EliminateDeadOp() : OptPass("EliminateDeadOp"){};
EliminateDeadOp::~EliminateDeadOp(){};

base::Status EliminateDeadOp::optimize(
    std::vector<TensorWrapper*>& tensor_repository,
    std::vector<OpWrapper*>& op_repository, int begin_op_index) {
  // 从后向前寻找无用的死节点
  int index = static_cast<int>(op_repository.size() - begin_op_index - 1);

  if (index < 0 || index >= op_repository.size()) {
    return base::kStatusCodeOk;
  }

  OpWrapper* op_wrapper = op_repository[index];

  bool use_flag = false;  // true表示是有用的节点

  //有后继节点，则是有用的节点
  if (op_wrapper->successors_.size() != 0) {
    use_flag = true;
  }

  // 没有后继节点，若其任一输出Tensor是Output Tensor,则是有用的节点
  if (!use_flag) {
    for (auto tensor_wrapper : tensor_repository) {
      auto it = std::find(tensor_wrapper->producers_.begin(),
                          tensor_wrapper->producers_.end(), op_wrapper);
      if (it != tensor_wrapper->producers_.end()) {
        if (tensor_wrapper->input_output_type_ == kOutput) {
          use_flag = true;
          break;
        }
      }
    }
  }

  // 消除这个无用节点
  if (!use_flag) {
    // 将其从前驱节点的后继节点中删除
    rmOpFromPredecessor(op_wrapper);

    // 处理被删除节点的输出Tensor
    // 若该Tensor仅有这一个生产者Op，则删除该Tensor
    // 若该Tensor其中一个生产者Op是该Op，则从其生产者Op中删除该Op

    rmOutputTensorAndMaybeDelete(op_wrapper, tensor_repository);

    // 处理被删除节点的输入Tensor
    // 若该Tensor仅有这一个消费者Op，则删除该Tensor
    // 若该Tensor其中一个消费者Op是该Op，则从其消费者Op中删除该Op
    rmInputTensorAndMaybeDelete(op_wrapper, tensor_repository);

    //将待删除Op从Op仓库删除
    auto it = std::find(op_repository.begin(), op_repository.end(), op_wrapper);
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

  if (use_flag) {
    return optimize(tensor_repository, op_repository, begin_op_index + 1);
  } else {
    // op_repository中的op数量减少了，begin_op_index不应该增加
    return optimize(tensor_repository, op_repository, begin_op_index);
  }
}

TypeOptPassRegister<TypeOptPassCreator<EliminateDeadOp>>
    g_eliminate_dead_op_register(base::kDeviceTypeCodeCpu,
                                 kOptPassTypeEliminateDeadOp, 3);

}  // namespace net

}  // namespace nndeploy