

#include "nndeploy/net/tensor_pool.h"

namespace nndeploy {
namespace net {

TensorPool::TensorPool(device::Device *device,
                       std::vector<TensorWrapper *> &tensor_repository,
                       std::vector<OpWrapper *> &op_repository)
    : device_(device),
      tensor_repository_(tensor_repository),
      op_repository_(op_repository) {}

TensorPool::~TensorPool() {}

void TensorPool::setAllocateInputOutputTensor(
    bool allocate_input_output_tensor) {
  allocate_input_output_tensor_ = allocate_input_output_tensor;
}

int64_t TensorPool::getMemorySize() {
  NNDEPLOY_LOGE("TensorPool::getMemorySize is not implemented!\n");
  return 0;
}
base::Status TensorPool::setMemory(device::Buffer *buffer) {
  NNDEPLOY_LOGE("TensorPool::setMemory is not implemented!\n");
  return base::kStatusCodeErrorNotImplement;
}

std::map<TensorPoolType, std::shared_ptr<TensorPoolCreator>>
    &getGlobalTensorPoolCreatorMap() {
  static std::once_flag once;
  static std::shared_ptr<
      std::map<TensorPoolType, std::shared_ptr<TensorPoolCreator>>>
      creators;
  std::call_once(once, []() {
    creators.reset(
        new std::map<TensorPoolType, std::shared_ptr<TensorPoolCreator>>);
  });
  return *creators;
}

TensorPool *createTensorPool(TensorPoolType type, device::Device *device,
                             std::vector<TensorWrapper *> &tensor_repository,
                             std::vector<OpWrapper *> &op_repository) {
  TensorPool *temp = nullptr;
  auto &creater_map = getGlobalTensorPoolCreatorMap();
  if (creater_map.count(type) != 0) {
    temp = creater_map[type]->createTensorPool(device, tensor_repository,
                                               op_repository);
  }
  return temp;
}

std::vector<int> getOpOrderIndex(std::vector<OpWrapper *> &producers,
                                 std::vector<OpWrapper *> &consumers,
                                 std::vector<OpWrapper *> &op_repository) {
  std::vector<int> order_index;

  for (size_t i = 0; i < op_repository.size(); i++) {
    for (size_t j = 0; j < producers.size(); j++) {
      if (op_repository[i] == producers[j]) {
        order_index.push_back(i);
        break;
      }
    }

    for (size_t j = 0; j < consumers.size(); j++) {
      if (op_repository[i] == consumers[j]) {
        order_index.push_back(i);
        break;
      }
    }
  }

  return order_index;
}

bool isInterval(std::array<int, 2> &interval,
                std::vector<std::array<int, 2>> &intervals) {
  for (size_t i = 0; i < intervals.size(); i++) {
    // 检查interval的起始点是否在intervals[i]内
    if (interval[0] >= intervals[i][0] && interval[0] <= intervals[i][1]) {
      return true;
    }
    // 检查interval的结束点是否在intervals[i]内
    if (interval[1] >= intervals[i][0] && interval[1] <= intervals[i][1]) {
      return true;
    }
    // 检查interval是否完全包含intervals[i]
    if (interval[0] <= intervals[i][0] && interval[1] >= intervals[i][1]) {
      return true;
    }
  }
  return false;
}

void tensorUsageRecordPrint(
    const std::vector<std::shared_ptr<TensorUsageRecord>>
        &tensor_usage_records) {
  // 统计tensor的个数，并累加大小
  size_t total_tensor_count = 0;
  size_t total_memory_size = 0;
  for (const auto &tensor_usage_record : tensor_usage_records) {
    total_tensor_count++;
    total_memory_size += tensor_usage_record->size_;
  }
  NNDEPLOY_LOGE("Total tensor count: %zu\n", total_tensor_count);
  NNDEPLOY_LOGE("Total memory size: %zu\n", total_memory_size);
}

void chunkPrint(const std::vector<std::shared_ptr<Chunk>> &chunks) {
  // 统计chunk的个数，并累加大小
  size_t total_chunk_count = 0;
  size_t total_chunk_size = 0;
  for (const auto &chunk : chunks) {
    total_chunk_count++;
    total_chunk_size += chunk->buffer_->getSize();
  }
  NNDEPLOY_LOGE("Total chunk count: %zu\n", total_chunk_count);
  NNDEPLOY_LOGE("Total chunk size: %zu\n", total_chunk_size);
}

}  // namespace net
}  // namespace nndeploy
