#include "nndeploy/net/optimizer/eliminate_common_subexpression.h"

#include <sstream>
#include <unordered_map>

namespace nndeploy {

namespace net {

/// https://stackoverflow.com/questions/2590677/how-do-i-combine-hash-values-in-c0x
inline void hash_combine(std::size_t& seed) {}

// 用于组合多个哈希值以生成一个单一的哈希值
template <typename Hasher, typename T, typename... Rest>
void hash_combine(std::size_t& seed, const Hasher& hasher, const T& v,
                  Rest... rest) {
  seed ^= hasher(v) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
  hash_combine(seed, rest...);
}

/**
 * 比较两个Tensor是否相等
 *
 */
bool compareTensor(const device::Tensor* lhs, const device::Tensor* rhs) {
  return lhs == rhs;
}

struct CSEOpHash {
  std::size_t operator()(const OpWrapper* op_wrapper) const {
    std::size_t seed = 0;
    auto size_t_hasher = std::hash<std::size_t>();
    auto string_hasher = std::hash<std::string>();

    // OpType
    hash_combine(seed, std::hash<uint32_t>(),
                 static_cast<uint32_t>(op_wrapper->op_->getOpType()));

    // DeviceType PrecisionType
    hash_combine(
        seed, std::hash<uint32_t>(),
        static_cast<uint32_t>(op_wrapper->op_->getDeviceType().code_),
        std::hash<uint32_t>(),
        static_cast<uint32_t>(op_wrapper->op_->getDeviceType().device_id_),
        std::hash<uint32_t>(),
        static_cast<uint32_t>(op_wrapper->op_->getPrecisionType()));

    // predecessor name
    hash_combine(seed, size_t_hasher, op_wrapper->predecessors_.size());
    for (auto predecessor : op_wrapper->predecessors_) {
      hash_combine(seed, string_hasher, predecessor->name_);
    }

    // input name
    hash_combine(seed, size_t_hasher,
                 op_wrapper->op_->getAllInputName().size());
    for (auto input_name : op_wrapper->op_->getAllInputName()) {
      hash_combine(seed, string_hasher, input_name);
    }

    // output name
    hash_combine(seed, size_t_hasher,
                 op_wrapper->op_->getAllOutputName().size());

    return seed;
  }
};

struct CSEOpEqual {
  bool operator()(const OpWrapper* lhs, const OpWrapper* rhs) const {
    if (!lhs) {
      return !rhs;
    } else if (!rhs) {
      return !lhs;
    }

    // 判断 OpType、DeviceType、PrecisionType
    if (lhs->op_->getOpType() != rhs->op_->getOpType() ||
        lhs->op_->getDeviceType() != rhs->op_->getDeviceType() ||
        lhs->op_->getPrecisionType() != rhs->op_->getPrecisionType()) {
      return false;
    }

    // 判断参数一致

    if (rhs->op_->getParam() != nullptr && lhs->op_->getParam() != nullptr) {
      std::ostringstream oss;
      rhs->op_->getParam()->serialize(oss);
      std::string param_r_str = oss.str();

      oss.str("");
      oss.clear();

      lhs->op_->getParam()->serialize(oss);
      std::string param_l_str = oss.str();

      oss.str("");
      oss.clear();

      if (param_l_str != param_r_str) {
        return false;
      }
    }

    // 判断输入一致
    std::vector<device::Tensor*> inputs_l = lhs->op_->getAllInput();
    std::vector<device::Tensor*> inputs_r = rhs->op_->getAllInput();
    if (inputs_l.size() != inputs_r.size()) {
      return false;
    }
    for (int i = 0; i < inputs_l.size(); i++) {
      if (!compareTensor(inputs_l[i], inputs_r[i])) {
        return false;
      }
    }

    // 判断输出个数一致
    std::vector<device::Tensor*> outputs_l = lhs->op_->getAllOutput();
    std::vector<device::Tensor*> outputs_r = rhs->op_->getAllOutput();
    if (outputs_l.size() != outputs_r.size()) {
      return false;
    }

    return true;
  }
};

EliminateCommonSubexpression::EliminateCommonSubexpression() {}

EliminateCommonSubexpression::~EliminateCommonSubexpression() {}

base::Status EliminateCommonSubexpression::optimize(
    std::vector<TensorWrapper*>& tensor_repository,
    std::vector<OpWrapper*>& op_repository, int begin_op_index) {
  std::unordered_map<OpWrapper*, OpWrapper*, CSEOpHash, CSEOpEqual>
      op_wrapper_map;

  // 使用hash_map 找到所有相同Op
  for (auto op_wrapper_it = op_repository.begin();
       op_wrapper_it != op_repository.end();) {
    auto op_wrapper = *op_wrapper_it;

    if (op_wrapper_map.find(op_wrapper) == op_wrapper_map.end()) {
      op_wrapper_map[op_wrapper] = op_wrapper;
      op_wrapper_it++;
    } else {
      std::set<TensorWrapper*> to_delete_tensors;

      // 要保留的公共Op
      auto reserved_op_wrapper = op_wrapper_map[op_wrapper];

      // 如果某个Tensor的消费者包含当前待删除的Op，则删去
      for (auto tensor_wrapper : tensor_repository) {
        auto it = std::find(tensor_wrapper->consumers_.begin(),
                            tensor_wrapper->consumers_.end(), op_wrapper);
        if (it != tensor_wrapper->consumers_.end()) {
          tensor_wrapper->consumers_.erase(it);
        }
      }

      // 当前待删除的Op的输出Tensor,
      // 如果作为后继Op的输入Tensor，则需要将其替换为被保留的op的输出Tensor
      std::vector<device::Tensor*> output_tensors =
          op_wrapper->op_->getAllOutput();
      std::vector<device::Tensor*> reserved_output_tensors =
          reserved_op_wrapper->op_->getAllOutput();

      for (auto successor : op_wrapper->successors_) {
        std::vector<device::Tensor*> success_input_tensors =
            successor->op_->getAllInput();
        for (int i = 0; i < output_tensors.size(); i++) {
          auto it = std::find(success_input_tensors.begin(),
                              success_input_tensors.end(), output_tensors[i]);
          if (it != success_input_tensors.end()) {
            // 用被保留的op的输出Tensor替换对应位置
            successor->op_->setInput(reserved_output_tensors[i],
                                     it - success_input_tensors.begin());
          }
        }
      }

      for (auto tensor_wrapper : tensor_repository) {
        auto prod_it = std::find(tensor_wrapper->producers_.begin(),
                                 tensor_wrapper->producers_.end(), op_wrapper);
        // 处理待删除op的输出Tensor
        // 表明待删除的Op是当前TensorWrapper的生产者，则删去当前TensorWrapper
        if (prod_it != tensor_wrapper->producers_.end()) {
          // 该TensorWrapper的生产者只有一个，为被删除的op
          // 删除该TensorWrapper
          if (tensor_wrapper->producers_.size() == 1) {
            to_delete_tensors.insert(tensor_wrapper);

          } else {
            // 该TensorWrapper的生产者有多个
            // 从生产者中删除 待删除的Opwrapper
            tensor_wrapper->producers_.erase(prod_it);
          }
        }

        // 处理待删除op的输入Tensor  同上
        auto cons_it = std::find(tensor_wrapper->consumers_.begin(),
                                 tensor_wrapper->consumers_.end(), op_wrapper);
        if (cons_it != tensor_wrapper->consumers_.end()) {
          if (tensor_wrapper->consumers_.size() == 1) {
            to_delete_tensors.insert(tensor_wrapper);
          } else {
            tensor_wrapper->consumers_.erase(cons_it);
          }
        }
      }

      // 从其前驱的后继中删除
      for (auto predecessor : op_wrapper->predecessors_) {
        predecessor->successors_.erase(
            std::remove(predecessor->successors_.begin(),
                        predecessor->successors_.end(), op_wrapper),
            predecessor->successors_.end());
      }

      // 将被删除的Op的后继的前驱改为被保留的Op
      for (auto successor : op_wrapper->successors_) {
        successor->predecessors_.erase(
            std::remove(successor->predecessors_.begin(),
                        successor->predecessors_.end(), op_wrapper),
            successor->predecessors_.end());
        successor->predecessors_.emplace_back(reserved_op_wrapper);
        reserved_op_wrapper->successors_.emplace_back(successor);
      }

      // 删除标记的TensorWrapper
      for (auto tensor_wrapper : to_delete_tensors) {
        if (tensor_wrapper->tensor_ != nullptr) {
          delete tensor_wrapper->tensor_;
          tensor_wrapper->tensor_ = nullptr;
        }

        NNDEPLOY_LOGE("delete tensor name: %s\n",
                      tensor_wrapper->name_.c_str());
        auto it = std::find(tensor_repository.begin(), tensor_repository.end(),
                            tensor_wrapper);
        if (it != tensor_repository.end()) {
          tensor_repository.erase(it);
        }

        delete tensor_wrapper;
      }

      auto it =
          std::find(op_repository.begin(), op_repository.end(), op_wrapper);
      if (it != op_repository.end()) {
        op_repository.erase(op_wrapper_it);
        NNDEPLOY_LOGE("delete op name: %s\n", op_wrapper->name_.c_str());
      }

      if (op_wrapper->op_ != nullptr) {
        delete op_wrapper->op_;
        op_wrapper->op_ = nullptr;
      }
      delete op_wrapper;
    }
  }

  return base::kStatusCodeOk;
}

TypeOptPassRegister<TypeOptPassCreator<EliminateCommonSubexpression>>
    g_eliminate_common_subexpression_register(
        base::kDeviceTypeCodeCpu, kOptPassTypeEliminateCommonSubexpression);

}  // namespace net

}  // namespace nndeploy