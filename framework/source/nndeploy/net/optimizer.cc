#include "nndeploy/net/optimizer.h"

#include "nndeploy/net/net.h"
namespace nndeploy {
namespace net {

// OptPass
OptPass::OptPass(std::string name) { name_ = name; }

OptPass::~OptPass() {}

std::string OptPass::getName() { return name_; }

base::Status OptPass::setNet(Net* net) {
  this->net_ = net;
  return base::kStatusCodeOk;
}

/**
 * @brief 模式匹配
 *
 * @param tensor_repository
 * @param op_repository
 * @param pass_types
 * @return op_repository中匹配到的首个op的index，如果未匹配到则返回-1
 * @note 匹配规则：
 * 1. 匹配到的op的类型为pass_types中的第一个，且op的successors_中有且只有一个
 * 2.
 * 该op的successors_的类型为pass_types中的第二个，且op的successors_的predecessors_中有且只有一个
 * 3. 以此类推，直到pass_types中的最后一个
 * 4.
 * TODO：除最后一个节点外，中间节点的输出tensor不能为模型的输出节点（该要点还未实现）
 */
int OptPass::seqPatternMatch(std::vector<TensorWrapper*>& tensor_repository,
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

        if (next_op->op_->getOpType() != types[j] ||
            next_op->predecessors_.size() != 1) {
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

/**
 * @brief 模式匹配并更新tensor_repository
 *
 * @param tensor_repository
 * @param op_repository
 * @param pass_types
 * @param begin_op_index
 * @return 是否成功
 * @note 更新策略
 * 1. 更新tensor_repository：
 *    a. 更新最后一个op的输出：其生产者改为第一个op
 *    b. 删除除开最后一个op以外所有的输出
 */
base::Status OptPass::seqPatternMatchUpateTensorRepository(
    std::vector<TensorWrapper*>& tensor_repository,
    std::vector<OpWrapper*>& op_repository,
    const std::vector<ir::OpType>& types, int begin_op_index) {
  if (begin_op_index < 0 || begin_op_index >= op_repository.size()) {
    NNDEPLOY_LOGE("begin_op_index[%d] is invalid!\n", begin_op_index);
    return base::kStatusCodeErrorInvalidParam;
  }

  OpWrapper* first_op = op_repository[begin_op_index];
  OpWrapper* last_op = first_op;

  // 找到最后一个op
  for (int i = 1; i < types.size(); ++i) {
    if (last_op->successors_.size() != 1) {
      return base::kStatusCodeErrorInvalidParam;
    }
    last_op = last_op->successors_[0];
  }

  // for (auto& tensor : tensor_repository) {
  //   NNDEPLOY_LOGE("Tensor name: %s\n", tensor->name_.c_str());
  // }

  // 删除除最后一个op以外所有的输出
  OpWrapper* current_op = first_op;
  while (current_op != last_op) {
    rmOutputTensorAndMaybeDelete(current_op, tensor_repository);
    current_op = current_op->successors_[0];
  }

  // 更新最后一个op的输出的生产者为第一个op
  for (auto& tensor : tensor_repository) {
    if (tensor->producers_.size() != 1) {
      continue;
    }
    if (tensor->producers_[0] == last_op) {
      tensor->producers_[0] = first_op;
    }
  }

  // for (auto& tensor : tensor_repository) {
  //   NNDEPLOY_LOGE("Tensor name: %s\n", tensor->name_.c_str());
  // }

  return base::kStatusCodeOk;
}

/*
 * @brief 更新op_repository
 * @param tensor_repository
 * @param op_repository
 * @return
 * @note
 * 1. 更新op_repository：
 *    a. 更新第一个节点：OpWrapper的successors_改为最后一个节点的successors_；
 *    b. 更新最后一个节点的successors_节点：该节点的前驱节点改为第一个节点
 *    c. 删除除第一个节点外的节点
 */
base::Status OptPass::seqPatternMatchUpateOpRepository(
    std::vector<TensorWrapper*>& tensor_repository,
    std::vector<OpWrapper*>& op_repository,
    const std::vector<ir::OpType>& types, int begin_op_index) {
  if (begin_op_index >= op_repository.size() || begin_op_index < 0) {
    return base::kStatusCodeErrorInvalidParam;
  }

  OpWrapper* first_op = op_repository[begin_op_index];
  OpWrapper* last_op = first_op;

  // 找到最后一个op
  for (int i = 1; i < types.size(); ++i) {
    if (last_op->successors_.empty()) {
      return base::kStatusCodeErrorInvalidParam;
    }
    last_op = last_op->successors_[0];
  }

  // NNDEPLOY_LOGE("第一个op的name: %s\n", first_op->name_.c_str());
  // NNDEPLOY_LOGE("最后一个op的name: %s\n", last_op->name_.c_str());

  // c. 删除除第一个节点外的节点
  OpWrapper* current_op = first_op->successors_[0];
  while (current_op != last_op) {
    NNDEPLOY_LOGE("current_op name: %s\n", current_op->name_.c_str());
    OpWrapper* next_op = current_op->successors_[0];
    // 从op_repository中删除current_op
    op_repository.erase(
        std::remove(op_repository.begin(), op_repository.end(), current_op),
        op_repository.end());
    if (current_op->op_ != nullptr) {
      delete current_op->op_;
    }
    delete current_op;
    current_op = next_op;
  }

  // a. 更新第一个节点：OpWrapper的successors_改为最后一个节点的successors_
  first_op->successors_ = last_op->successors_;
  // NNDEPLOY_LOGE("first_op name: %s\n", first_op->name_.c_str());

  // b. 更新最后一个节点的successors_节点：该节点的前驱节点改为第一个节点
  for (auto successor : last_op->successors_) {
    for (auto& predecessor : successor->predecessors_) {
      if (predecessor == last_op) {
        predecessor = first_op;
      }
    }
  }

  // 删除最后一个节点
  op_repository.erase(
      std::remove(op_repository.begin(), op_repository.end(), last_op),
      op_repository.end());
  if (last_op->op_ != nullptr) {
    delete last_op->op_;
  }
  delete last_op;

  //

  return base::kStatusCodeOk;
}

base::Status OptPass::rmOpFromPredecessor(OpWrapper* op_wrapper) {
  for (auto predecessor : op_wrapper->predecessors_) {
    auto it = std::find(predecessor->successors_.begin(),
                        predecessor->successors_.end(), op_wrapper);
    if (it != predecessor->successors_.end()) {
      predecessor->successors_.erase(it);
    }
  }
  return base::kStatusCodeOk;
}

base::Status OptPass::rmOpFromSuccessors(OpWrapper* op_wrapper) {
  for (auto successor : op_wrapper->successors_) {
    auto it = std::find(successor->predecessors_.begin(),
                        successor->predecessors_.end(), op_wrapper);
    if (it != successor->predecessors_.end()) {
      successor->predecessors_.erase(it);
    }
  }
  return base::kStatusCodeOk;
}

base::Status OptPass::rmOutputTensorAndMaybeDelete(
    OpWrapper* op_wrapper, std::vector<TensorWrapper*>& tensor_repository) {
  std::set<TensorWrapper*> to_delete_tensors;

  for (auto tensor_wrapper : tensor_repository) {
    auto prod_it = std::find(tensor_wrapper->producers_.begin(),
                             tensor_wrapper->producers_.end(), op_wrapper);
    if (prod_it != tensor_wrapper->producers_.end()) {
      if (tensor_wrapper->producers_.size() == 1) {
        // 仅有这一个生产者Op，则直接释放这个Op
        to_delete_tensors.insert(tensor_wrapper);
      } else {
        tensor_wrapper->producers_.erase(prod_it);
      }
    }
  }

  for (auto tensor_wrapper : to_delete_tensors) {
    if (tensor_wrapper->tensor_ != nullptr) {
      // 删除 Net中的输入tensor
      // 在图优化时，部分Tensor被释放，需要删除Net中的对应tensor
      net_->rmInput(tensor_wrapper->tensor_);
      delete tensor_wrapper->tensor_;
      tensor_wrapper->tensor_ = nullptr;
    }
    // NNDEPLOY_LOGE("delete tensor name: %s\n", tensor_wrapper->name_.c_str());
    auto it = std::find(tensor_repository.begin(), tensor_repository.end(),
                        tensor_wrapper);
    if (it != tensor_repository.end()) {
      tensor_repository.erase(it);
    }
    delete tensor_wrapper;
  }
  return base::kStatusCodeOk;
}

base::Status OptPass::rmInputTensorAndMaybeDelete(
    OpWrapper* op_wrapper, std::vector<TensorWrapper*>& tensor_repository) {
  std::set<TensorWrapper*> to_delete_tensors;
  for (auto tensor_wrapper : tensor_repository) {
    auto cons_it = std::find(tensor_wrapper->consumers_.begin(),
                             tensor_wrapper->consumers_.end(), op_wrapper);
    if (cons_it != tensor_wrapper->consumers_.end()) {
      if (tensor_wrapper->consumers_.size() == 1) {
        // 仅有这一个消费者Op，则直接释放这个Op
        to_delete_tensors.insert(tensor_wrapper);
      } else {
        tensor_wrapper->consumers_.erase(cons_it);
      }
    }
  }

  for (auto tensor_wrapper : to_delete_tensors) {
    if (tensor_wrapper->tensor_ != nullptr) {
      // 删除 Net中的输入tensor
      // 在图优化时，部分Tensor被释放，需要删除Net中的对应tensor
      net_->rmInput(tensor_wrapper->tensor_);
      delete tensor_wrapper->tensor_;
      tensor_wrapper->tensor_ = nullptr;
    }
    // NNDEPLOY_LOGE("delete tensor name: %s\n", tensor_wrapper->name_.c_str());
    auto it = std::find(tensor_repository.begin(), tensor_repository.end(),
                        tensor_wrapper);
    if (it != tensor_repository.end()) {
      tensor_repository.erase(it);
    }
    delete tensor_wrapper;
  }
  return base::kStatusCodeOk;
}

// 工厂模式
// 设备类型  ->  优化等级  -> Pass类型
std::map<base::DeviceTypeCode,
         std::map<int, std::map<OptPassType, std::shared_ptr<OptPassCreator>>>>&
getGlobalOptPassCreatorMap() {
  static std::once_flag once;
  static std::shared_ptr<std::map<
      base::DeviceTypeCode,
      std::map<int, std::map<OptPassType, std::shared_ptr<OptPassCreator>>>>>
      creators;
  std::call_once(once, []() {
    creators.reset(
        new std::map<base::DeviceTypeCode,
                     std::map<int, std::map<OptPassType,
                                            std::shared_ptr<OptPassCreator>>>>);
  });
  return *creators;
}

std::shared_ptr<OptPass> createOptPass(base::DeviceType device_type, int level,
                                       OptPassType type) {
  auto& creator_map = getGlobalOptPassCreatorMap();
  auto device_map = creator_map.find(device_type.code_);
  if (device_map != creator_map.end()) {
    auto level_map = device_map->second.find(level);
    if (level_map != device_map->second.end()) {
      auto pass_creator = level_map->second.find(type);
      if (pass_creator != level_map->second.end()) {
        return pass_creator->second->createOptPass();
      }
    }
  }
  return nullptr;
}

// Optimizer
Optimizer::Optimizer() {}

Optimizer::~Optimizer() {}

base::Status Optimizer::init(base::DeviceType device_type,
                             std::set<OptPassType> enable_pass,
                             std::set<OptPassType> disable_pass) {
  device_type_ = device_type;
  auto& creator_map = getGlobalOptPassCreatorMap();
  auto device_map = creator_map.find(device_type.code_);
  if (device_map != creator_map.end()) {
    //根据Pass优先级进行图优化
    for (auto level_map : device_map->second) {
      for (auto& pass_creator : level_map.second) {
        // 设置了仅启用某些pass，当前pass不在的话则跳过
        if (!enable_pass.empty() &&
            enable_pass.find(pass_creator.first) == enable_pass.end()) {
          continue;
        }

        // 设置了禁用某些pass，当前pass在的话则跳过
        if (enable_pass.empty() && !disable_pass.empty() &&
            disable_pass.find(pass_creator.first) != disable_pass.end()) {
          continue;
        }
        if (opt_passes_.find(level_map.first) == opt_passes_.end()) {
          opt_passes_[level_map.first] =
              std::map<OptPassType, std::shared_ptr<OptPass>>();
        }
        opt_passes_[level_map.first][pass_creator.first] =
            pass_creator.second->createOptPass();
      }
    }
  }
  return base::kStatusCodeOk;
}
base::Status Optimizer::deinit() {
  opt_passes_.clear();
  return base::kStatusCodeOk;
}
base::Status Optimizer::addPass(OptPassType type, int level) {
  if (opt_passes_.find(type) == opt_passes_.end()) {
    if (opt_passes_.find(level) == opt_passes_.end()) {
      opt_passes_[level] = std::map<OptPassType, std::shared_ptr<OptPass>>();
    }
    opt_passes_[level][type] = createOptPass(device_type_, level, type);
  }
  return base::kStatusCodeOk;
}
base::Status Optimizer::removePass(OptPassType type) {
  // 遍历外层 map
  for (auto& outer_pair : opt_passes_) {
    // 在内层 map 中找到对应的 OptPassType
    auto& inner_map = outer_pair.second;
    auto it = inner_map.find(type);
    if (it != inner_map.end()) {
      // 找到了，从内层 map 中删除
      inner_map.erase(it);
      // 如果内层 map 为空，从外层 map 删除这个元素
      if (inner_map.empty()) {
        opt_passes_.erase(outer_pair.first);
      }
    }
  }
  return base::kStatusCodeOk;
}
base::Status Optimizer::optimize(std::vector<TensorWrapper*>& tensor_repository,
                                 std::vector<OpWrapper*>& op_repository,
                                 Net* net) {
  base::Status status = base::kStatusCodeOk;
  for (auto& level_map : opt_passes_) {
    for (auto& pass : level_map.second) {
      NNDEPLOY_LOGE("Execute pass: %s\n", pass.second->getName().c_str());
      pass.second->setNet(net);
      status = pass.second->optimize(tensor_repository, op_repository, 0);
      if (status != base::kStatusCodeOk) {
        NNDEPLOY_LOGE("optimize failed!\n");
        return status;
      }
    }
  }
  return base::kStatusCodeOk;
}

}  // namespace net
}  // namespace nndeploy
