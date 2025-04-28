#include "nndeploy/net/tensor_pool/tensor_pool_1d_offset_calculate_by_breadth.h"

#include "nndeploy/net/tensor_pool.h"

namespace nndeploy {
namespace net {

// tensorpool OffsetCalculateGreedyByBreadth 实现

TypeTensorPoolRegister<
    TypeTensorPoolCreator<TensorPool1DOffsetCalculateGreedyByBreadth>>
    g_tensor_pool_1d_offset_calculate_greedy_by_breadth_register(
        kTensorPool1DOffsetCalculateTypeGreedyByBreadth);

TensorPool1DOffsetCalculateGreedyByBreadth::
    TensorPool1DOffsetCalculateGreedyByBreadth(
        device::Device *device, std::vector<TensorWrapper *> &tensor_repository,
        std::vector<OpWrapper *> &op_repository)
    : TensorPool1D(device, tensor_repository, op_repository) {}

TensorPool1DOffsetCalculateGreedyByBreadth::
    ~TensorPool1DOffsetCalculateGreedyByBreadth() {}

base::Status TensorPool1DOffsetCalculateGreedyByBreadth::allocate() {
  base::Status status = base::kStatusCodeOk;

  // 会内存泄漏
  total_consumption_ = this->getMemorySize();
  // 分配内存
  if (is_external_ == false) {
    mem_block_ = new device::Buffer(device_, total_consumption_);
    uint8_t *data_ptr = (uint8_t *)mem_block_->getData();
    for (auto &t : tensor_usage_records_) {
      device::Buffer *buffer =
          new device::Buffer(device_, t->size_, data_ptr + t->offset_);
      t->tensor_wrapper_->tensor_->justModify(buffer, false);
    }
  }
  tensorUsageRecordPrint(tensor_usage_records_);
  NNDEPLOY_LOGE("Total memory size: %zu (OffSetByBreadth)\n",
                total_consumption_);

  return status;
}

base::Status TensorPool1DOffsetCalculateGreedyByBreadth::deallocate() {
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

  status = deinitOpBreadth();
  if (status != base::kStatusCodeOk) {
    NNDEPLOY_LOGE("deinitOpBreadth failed\n");
    return status;
  }

  status = deinitTensorUsageRecord();
  if (status != base::kStatusCodeOk) {
    NNDEPLOY_LOGE("deinitTensorUsageRecord failed\n");
    return status;
  }

  return status;
}

int64_t TensorPool1DOffsetCalculateGreedyByBreadth::getMemorySize() {
  base::Status status = base::kStatusCodeOk;

  if (total_consumption_ != -1) {
    return total_consumption_;
  }

  // 初始化TensorUsageRecord, 对tensor大小进行排序
  status = initTensorUsageRecord();
  if (status != base::kStatusCodeOk) {
    NNDEPLOY_LOGE("initTensorUsageRecord failed\n");
    return -1;
  }

  // 初始化OpBreadth
  status = initOpBreadth();
  if (status != base::kStatusCodeOk) {
    NNDEPLOY_LOGE("initOpBreadth failed\n");
    return -1;
  }

  // 遍历每个张量使用记录
  for (auto &task : op_breadths_) {
    for (auto &t : task->breadth_) {
      if (assigned_tensors_.count(t) != 0) {
        continue;
      }
      tensor_num_++;
      std::shared_ptr<Offset> best_offset = nullptr;
      int smallest_gap = INT_MAX;  // 最小间隙

      // 遍历已分配的张量，寻找最佳偏移量
      for (int i = 0; i < offsets_.size(); i++) {
        size_t offset_size = offsets_[i]->size_;
        size_t tensor_size = t->size_;
        if (offset_size >= tensor_size) {
          std::vector<std::array<int, 2>> intervals;
          // 逐个判断优化
          for (const auto &y : offsets_[i]->tensor_usage_records_) {
            intervals.push_back(y->interval_);
          }
          if (isInterval(t->interval_, intervals)) {
            continue;
          }
          if (smallest_gap > offset_size - tensor_size) {
            best_offset = offsets_[i];
            smallest_gap = offset_size - tensor_size;
          }
        }
      }
      if (best_offset == nullptr) {
        best_offset = std::make_shared<Offset>();
        best_offset->offset_ = total_consumption_;
        best_offset->size_ = t->size_;
        // best_offset->tensor_usage_records_.push_back(t);
        offsets_.push_back(best_offset);
        total_consumption_ += t->size_;
      }
      best_offset->tensor_usage_records_.push_back(t);
      t->is_allocated_ = true;
      t->offset_ = best_offset->offset_;
      assigned_tensors_.insert(t);
    }
  }
  return total_consumption_;
}

base::Status TensorPool1DOffsetCalculateGreedyByBreadth::setMemory(
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
  uint8_t *data_ptr = (uint8_t *)mem_block_->getData();
  for (auto &t : tensor_usage_records_) {
    device::Buffer *buffer =
        new device::Buffer(device_, t->size_, data_ptr + t->offset_);
    t->tensor_wrapper_->tensor_->justModify(buffer, false);
  }
  return base::kStatusCodeOk;
}

}  // namespace net
}  // namespace nndeploy