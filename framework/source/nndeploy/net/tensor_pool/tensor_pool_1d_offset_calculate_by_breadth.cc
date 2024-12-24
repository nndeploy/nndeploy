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

  // // 初始化TensorUsageRecord, 对tensor大小进行排序
  // status = initTensorUsageRecord();
  // if (status != base::kStatusCodeOk) {
  //   NNDEPLOY_LOGE("initTensorUsageRecord failed\n");
  //   return status;
  // }

  // // 初始化OpBreadth
  // status = initOpBreadth();
  // if (status != base::kStatusCodeOk) {
  //   NNDEPLOY_LOGE("initOpBreadth failed\n");
  //   return status;
  // }

  // // 记录内存块大小
  // size_t total_consumption_ = 0;
  // int tensor_num_ = 0;
  // // 遍历每个张量使用记录
  // for (auto& task : op_breadths_) {
  //   for (auto& t : task->breadth_) {
  //       auto it = std::find(ordered_allocated_ids_.begin(),
  //       ordered_allocated_ids_.end(), t); if (it !=
  //       ordered_allocated_ids_.end()) continue; tensor_num_++; int
  //       prev_offset = 0; // 上一个张量的偏移量 int best_offset = -1; //
  //       最佳偏移量 int smallest_gap = INT_MAX; // 最小间隙

  //       // 遍历已分配的张量，寻找最佳偏移量
  //       for (const auto& x : ordered_allocated_ids_) {
  //           int max_first_op = std::max(t->interval_[0], x->interval_[0]);
  //           int min_last_op = std::min(t->interval_[1], x->interval_[1]);
  //           // 检查张量是否可以在当前间隙中分配
  //           if (max_first_op <= min_last_op) {
  //             int gap = x->offset_ - prev_offset; // 计算间隙大小
  //             // 如果间隙足够且比当前找到的最小间隙小，则更新最佳偏移量
  //             if (gap >= t->size_ && gap < smallest_gap) {
  //                 smallest_gap = gap;
  //                 best_offset = prev_offset;
  //             }
  //             prev_offset = std::max(prev_offset, x->offset_ +
  //             static_cast<int>(x->size_));
  //           }
  //           // 更新上一个张量的偏移量

  //       }
  //       if (best_offset==-1) best_offset = prev_offset;
  //       t->offset_ = best_offset;
  //       ordered_allocated_ids_.push_back(t);
  //       total_consumption_ = std::max(static_cast<int>(total_consumption_),
  //       best_offset + static_cast<int>(t->size_));
  //   }
  // }

  // 分配内存
  //  会内存泄漏
  //  device::Buffer *mem_block = new device::Buffer(device_,
  //  total_consumption_);
  total_consumption_ = this->getMemorySize();
  // 分配内存
  if (is_external_ == false) {
    mem_block_ = new device::Buffer(device_, total_consumption_);
  }
  uint8_t *data_ptr = (uint8_t *)mem_block_->getData();
  for (auto &t : tensor_usage_records_) {
    device::Buffer *buffer =
        new device::Buffer(device_, t->size_, data_ptr + t->offset_);
    device::TensorDesc tensor_desc = t->tensor_wrapper_->tensor_->getDesc();
    device::BufferDesc buffer_desc =
        device_->toBufferDesc(tensor_desc, base::IntVector());
    if (!buffer->justModify(buffer_desc)) {
      NNDEPLOY_LOGE("tensor name = %s.\n", t->tensor_wrapper_->name_.c_str());
      NNDEPLOY_LOGE("buffer->justModify failed\n");
      return base::kStatusCodeErrorInvalidValue;
    }
    t->tensor_wrapper_->tensor_->justModify(buffer, false);
  }
  tensorUsageRecordPrint(tensor_usage_records_);
  NNDEPLOY_LOGE("Total tensor num: %d \n", tensor_num_);
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

  // 记录内存块大小
  // size_t total_consumption_ = 0;
  // int tensor_num_ = 0;
  // 遍历每个张量使用记录
  for (auto &task : op_breadths_) {
    for (auto &t : task->breadth_) {
      auto it = std::find(ordered_allocated_ids_.begin(),
                          ordered_allocated_ids_.end(), t);
      if (it != ordered_allocated_ids_.end()) continue;
      tensor_num_++;
      int prev_offset = 0;         // 上一个张量的偏移量
      int best_offset = -1;        // 最佳偏移量
      int smallest_gap = INT_MAX;  // 最小间隙

      // 遍历已分配的张量，寻找最佳偏移量
      for (const auto &x : ordered_allocated_ids_) {
        int max_first_op = std::max(t->interval_[0], x->interval_[0]);
        int min_last_op = std::min(t->interval_[1], x->interval_[1]);
        // 检查张量是否可以在当前间隙中分配
        if (max_first_op <= min_last_op) {
          int gap = x->offset_ - prev_offset;  // 计算间隙大小
          // 如果间隙足够且比当前找到的最小间隙小，则更新最佳偏移量
          if (gap >= t->size_ && gap < smallest_gap) {
            smallest_gap = gap;
            best_offset = prev_offset;
          }
          prev_offset =
              std::max(prev_offset, x->offset_ + static_cast<int>(x->size_));
        }
        // 更新上一个张量的偏移量
      }
      if (best_offset == -1) best_offset = prev_offset;
      t->offset_ = best_offset;
      ordered_allocated_ids_.push_back(t);
      total_consumption_ = std::max(static_cast<int>(total_consumption_),
                                    best_offset + static_cast<int>(t->size_));
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
  return base::kStatusCodeOk;
}

}  // namespace net
}  // namespace nndeploy