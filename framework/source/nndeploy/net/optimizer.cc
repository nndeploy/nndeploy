#include "nndeploy/net/optimizer.h"

namespace nndeploy {
namespace net {

// OptPass
OptPass::OptPass() {}

OptPass::~OptPass() {}

/**
* @brief 模式匹配
* 
* @param tensor_repository 
* @param op_repository 
* @param pass_types 
* @return op_repository中匹配到的首个op的index，如果未匹配到则返回-1
* @note 匹配规则：
* 1. 匹配到的op的类型为pass_types中的第一个，且op的successors_中有且只有一个
* 2. 该op的successors_的类型为pass_types中的第二个，且op的successors_的predecessors_中有且只有一个
* 3. 以此类推，直到pass_types中的最后一个
*/
int OptPass::seqPatternMatch(std::vector<TensorWrapper *>& tensor_repository,
                  std::vector<OpWrapper *>& op_repository, const std::vector<ir::OpType>& types, int begin_op_index) {
  for (int i = begin_op_index; i < op_repository.size(); ++i) {
    OpWrapper* current_op = op_repository[i];
    if (current_op->op_->getOpType() == types[0] && current_op->successors_.size() == 1) {
      bool match = true;
      OpWrapper* next_op = current_op;
      
      for (int j = 1; j < types.size(); ++j) {
        next_op = next_op->successors_[0];
        
        if (next_op->op_->getOpType() != types[j] || next_op->predecessors_.size() != 1) {
          match = false;
          break;
        }
      }
      
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
base::Status OptPass::seqPatternMatchUpateTensorRepository(std::vector<TensorWrapper *>& tensor_repository,
                                std::vector<OpWrapper *>& op_repository, const std::vector<ir::OpType>& types, int begin_op_index) {
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

  // 更新最后一个op的输出的生产者为第一个op
  for (auto& tensor : tensor_repository) {
    if (tensor->producers_.size() != 1) {
      continue;
    }
    if (tensor->producers_[0] == last_op) {
      tensor->producers_[0] = first_op;
    }
  }

  // 删除除最后一个op以外所有的输出
  OpWrapper* current_op = first_op;
  while (current_op != last_op) {
    for (TensorWrapper*  tensor : tensor_repository) {
      if (tensor->producers_.size() != 1) {
        continue;
      }
      if (tensor->producers_[0] == current_op) {
        // 手动删除成员变量tensor_
        if (tensor->tensor_ != nullptr) {
          delete tensor->tensor_;
          tensor->tensor_ = nullptr;
        }
        // 从vector移除该tensor
        auto it = std::find(tensor_repository.begin(), tensor_repository.end(), tensor);
        if (it != tensor_repository.end()) {
          // NNDEPLOY_LOGE("delete tensor name: %s\n", tensor->name_.c_str());
          tensor_repository.erase(it);
        }
        // tensor_repository.erase(tensor);
        // 删除TensorWrapper对象
        delete tensor;
      }
    }
    current_op = current_op->successors_[0];
  }

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
base::Status OptPass::seqPatternMatchUpateOpRepository(std::vector<TensorWrapper *>& tensor_repository,
                                std::vector<OpWrapper *>& op_repository, const std::vector<ir::OpType>& types, int begin_op_index) {
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
    op_repository.erase(std::remove(op_repository.begin(), op_repository.end(), current_op), op_repository.end());
    if (current_op->op_ != nullptr) {
      delete current_op->op_;
    }
    delete current_op;
    current_op = next_op;
  }

  

  // a. 更新第一个节点：OpWrapper的successors_改为最后一个节点的successors_
  first_op->successors_ = last_op->successors_;
  NNDEPLOY_LOGE("first_op name: %s\n", first_op->name_.c_str());
  

  

  // b. 更新最后一个节点的successors_节点：该节点的前驱节点改为第一个节点
  for (auto successor : last_op->successors_) {
    for (auto& predecessor : successor->predecessors_) {
      if (predecessor == last_op) {
        predecessor = first_op;
      }
    }
  }

  

  // 删除最后一个节点
  op_repository.erase(std::remove(op_repository.begin(), op_repository.end(), last_op), op_repository.end());
  if (last_op->op_ != nullptr) {
      delete last_op->op_;
  }
  delete last_op;

  
  // 

  return base::kStatusCodeOk;
}

// 工厂模式
std::map<base::DeviceTypeCode, std::map<OptPassType, std::shared_ptr<OptPassCreator>>> &
getGlobalOptPassCreatorMap() {
  static std::once_flag once;
  static std::shared_ptr<
      std::map<base::DeviceTypeCode, std::map<OptPassType, std::shared_ptr<OptPassCreator>>>>
      creators;
  std::call_once(once, []() {
    creators.reset(
        new std::map<base::DeviceTypeCode, std::map<OptPassType, std::shared_ptr<OptPassCreator>>>);
  });
  return *creators;
}

std::shared_ptr<OptPass> createOptPass(base::DeviceType device_type, OptPassType type) {
  auto &creator_map = getGlobalOptPassCreatorMap();
  auto device_map = creator_map.find(device_type.code_);
  if (device_map != creator_map.end()) {
    auto pass_creator = device_map->second.find(type);
    if (pass_creator != device_map->second.end()) {
      return pass_creator->second->createOptPass();
    }
  }
  return nullptr;
}

// Optimizer
Optimizer::Optimizer() {}

Optimizer::~Optimizer() {}

base::Status Optimizer::init(base::DeviceType device_type){
  device_type_ = device_type;
  auto &creator_map = getGlobalOptPassCreatorMap();
  auto device_map = creator_map.find(device_type.code_);
  if (device_map != creator_map.end()) {
    for (auto &pass_creator : device_map->second) {
      opt_passes_[pass_creator.first] = pass_creator.second->createOptPass();
    }
  }
  return base::kStatusCodeOk;
}
base::Status Optimizer::deinit(){
  opt_passes_.clear();
  return base::kStatusCodeOk;
}
base::Status Optimizer::addPass(OptPassType type){
  if (opt_passes_.find(type) == opt_passes_.end()) {  
    opt_passes_[type] = createOptPass(device_type_, type);
  }
  return base::kStatusCodeOk;
}
base::Status Optimizer::removePass(OptPassType type){
  if (opt_passes_.find(type) != opt_passes_.end()) {
    opt_passes_.erase(type);
  }
  return base::kStatusCodeOk;
}
base::Status Optimizer::optimize(std::vector<TensorWrapper *>& tensor_repository,
                      std::vector<OpWrapper *>& op_repository) {
  base::Status status = base::kStatusCodeOk;  
  for (auto& pass : opt_passes_) {
    status = pass.second->optimize(tensor_repository, op_repository, 0);
    if (status != base::kStatusCodeOk) {
      NNDEPLOY_LOGE("optimize failed!\n");
      return status;
    }
  }
  return base::kStatusCodeOk;
}

}  // namespace net
}  // namespace nndeploy
