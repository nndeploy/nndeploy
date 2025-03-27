#include "nndeploy/net/tensor_pool/tensor_pool_1d_offset_calculate_by_size.h"

#include "nndeploy/net/tensor_pool.h"

namespace nndeploy {
namespace net {

// tensorpool OffsetCalculateGreedyBySize 实现

TypeTensorPoolRegister<
    TypeTensorPoolCreator<TensorPool1DOffsetCalculateGreedyBySize>>
    g_tensor_pool_1d_offset_calculate_greedy_by_size_register(
        kTensorPool1DOffsetCalculateTypeGreedyBySize);

TensorPool1DOffsetCalculateGreedyBySize::
    TensorPool1DOffsetCalculateGreedyBySize(
        device::Device *device, std::vector<TensorWrapper *> &tensor_repository,
        std::vector<OpWrapper *> &op_repository)
    : TensorPool1D(device, tensor_repository, op_repository) {}

TensorPool1DOffsetCalculateGreedyBySize::
    ~TensorPool1DOffsetCalculateGreedyBySize() {}

base::Status TensorPool1DOffsetCalculateGreedyBySize::allocate() {
  base::Status status = base::kStatusCodeOk;
  total_consumption_ = this->getMemorySize();
  // 分配内存
  if (is_external_ == false) {
    mem_block_ = new device::Buffer(device_, total_consumption_);
  }
  uint8_t *data_ptr = (uint8_t *)mem_block_->getData();
  for (auto &t : tensor_usage_records_) {
    device::Buffer *buffer =
        new device::Buffer(device_, t->size_, data_ptr + t->offset_);
    t->tensor_wrapper_->tensor_->justModify(buffer, false);
  }

  tensorUsageRecordPrint(tensor_usage_records_);
  NNDEPLOY_LOGE("Total memory size: %zu (OffSetBySize)\n", total_consumption_);

  return status;
}
base::Status TensorPool1DOffsetCalculateGreedyBySize::deallocate() {
  base::Status status = base::kStatusCodeOk;

  for (auto tensor_wrapper : tensor_repository_) {
    auto tensor = tensor_wrapper->tensor_;
    tensor->deallocate();
  }

  // @Leonisux 需要释放内存
  if (mem_block_ != nullptr && is_external_ == false) {
    delete mem_block_;
    mem_block_ = nullptr;
  }

  status = deinitTensorUsageRecord();
  if (status != base::kStatusCodeOk) {
    NNDEPLOY_LOGE("deinitTensorUsageRecord failed\n");
    return status;
  }

  return status;
}

int64_t TensorPool1DOffsetCalculateGreedyBySize::getMemorySize() {
  base::Status status = base::kStatusCodeOk;

  // 初始化TensorUsageRecord, 对tensor大小进行排序
  status = initTensorUsageRecord();
  if (status != base::kStatusCodeOk) {
    NNDEPLOY_LOGE("initTensorUsageRecord failed\n");
    return -1;
  }

  // 遍历每个张量使用记录
  for (auto &t : tensor_usage_records_) {
    if (t->is_allocated_) {
      continue;
    }

    // 寻找最佳偏移量
    std::shared_ptr<Offset> offset = nullptr;
    int smallest_gap = INT_MAX;  // 最小间隙，存在继续优化的空间

    for (int j = offsets_.size() - 1; j >= 0; j--) {
      size_t offset_size = offsets_[j]->size_;
      size_t tensor_size = t->size_;
      if (offset_size >= tensor_size) {
        std::vector<std::array<int, 2>> intervals;
        // 逐个判断优化
        for (const auto &y : offsets_[j]->tensor_usage_records_) {
          intervals.push_back(y->interval_);
        }
        bool flag = isInterval(t->interval_, intervals);
        if (!flag) {
          offset = offsets_[j];
          offset->tensor_usage_records_.push_back(t);
          break;
        }
      }
    }
    if (offset == nullptr) {
      offset = std::make_shared<Offset>();
      offset->offset_ = total_consumption_;
      offset->size_ = t->size_;
      offset->tensor_usage_records_.push_back(t);
      offsets_.push_back(offset);
      total_consumption_ += t->size_;
    }
    t->is_allocated_ = true;
    t->offset_ = offset->offset_;
  }
  return total_consumption_;
}

base::Status TensorPool1DOffsetCalculateGreedyBySize::setMemory(
    device::Buffer *buffer) {
  if (mem_block_ != nullptr && is_external_ == false) {
    delete mem_block_;
    mem_block_ = nullptr;
  }
  if (buffer == nullptr) {
    NNDEPLOY_LOGE("buffer is nullptr\n");
    return base::kStatusCodeErrorInvalidValue;
  }
  if (buffer->getRealSize() < total_consumption_) {
    NNDEPLOY_LOGE("buffer size is too small\n");
    return base::kStatusCodeErrorInvalidValue;
  }
  is_external_ = true;
  mem_block_ = buffer;
  return base::kStatusCodeOk;
}

}  // namespace net
}  // namespace nndeploy
