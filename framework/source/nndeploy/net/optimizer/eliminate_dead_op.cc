#include "nndeploy/net/optimizer/eliminate_dead_op.h"

namespace nndeploy {

namespace net {
EliminateDeadOp::EliminateDeadOp(){};
EliminateDeadOp::~EliminateDeadOp(){};

base::Status EliminateDeadOp::optimize(
    std::vector<TensorWrapper*>& tensor_repository,
    std::vector<OpWrapper*>& op_repository, int begin_op_index) {
  // 从后向前寻找无用的死节点
  int index = op_repository.size() - begin_op_index - 1;

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
    std::vector<TensorWrapper*> to_delete_tensors;

    // 将其从前驱节点的后继节点中删除
    for (auto predecessor : op_wrapper->predecessors_) {
      auto it = std::find(predecessor->successors_.begin(),
                          predecessor->successors_.end(), op_wrapper);
      if (it != predecessor->successors_.end()) {
        predecessor->successors_.erase(it);
      }
    }

    // 处理被删除节点的输入Tensor
    // 若该Tensor仅有这一个消费者Op，则删除该Tensor
    // 若该Tensor其中一个消费者Op是该Op，则从其消费者Op中删除该Op
    for (auto tensor_wrapper : tensor_repository) {
      auto prod_it = std::find(tensor_wrapper->producers_.begin(),
                               tensor_wrapper->producers_.end(), op_wrapper);
      if (prod_it != tensor_wrapper->producers_.end()) {
        if (tensor_wrapper->producers_.size() == 1) {
          to_delete_tensors.push_back(tensor_wrapper);
        } else {
          tensor_wrapper->producers_.erase(prod_it);
        }
      }
    }

    // 处理被删除节点的输出Tensor
    // 若该Tensor仅有这一个生产者Op，则删除该Tensor
    // 若该Tensor其中一个生产者Op是该Op，则从其生产者Op中删除该Op
    for (auto tensor_wrapper : tensor_repository) {
      auto cons_it = std::find(tensor_wrapper->consumers_.begin(),
                               tensor_wrapper->consumers_.end(), op_wrapper);
      if (cons_it != tensor_wrapper->consumers_.end()) {
        if (tensor_wrapper->consumers_.size() == 1) {
          to_delete_tensors.push_back(tensor_wrapper);
        } else {
          tensor_wrapper->consumers_.erase(cons_it);
        }
      }
    }

    // 删除标记的TensorWrapper
    for (auto tensor_wrapper : to_delete_tensors) {
      if (tensor_wrapper->tensor_ != nullptr) {
        delete tensor_wrapper->tensor_;
        tensor_wrapper->tensor_ = nullptr;
      }
      NNDEPLOY_LOGE("delete tensor name: %s\n", tensor_wrapper->name_.c_str());
      auto it = std::find(tensor_repository.begin(), tensor_repository.end(),
                          tensor_wrapper);
      if (it != tensor_repository.end()) {
        tensor_repository.erase(it);
      }
      delete tensor_wrapper;
    }

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
                                 kOptPassTypeEliminateDeadOp);

}  // namespace net

}  // namespace nndeploy