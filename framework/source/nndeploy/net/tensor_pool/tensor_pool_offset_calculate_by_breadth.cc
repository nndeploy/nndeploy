#include "nndeploy/net/tensor_pool/tensor_pool_offset_calculate_by_breadth.h"

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

  // 初始化TensorUsageRecord, 对tensor大小进行排序
  status = initTensorUsageRecord();
  if (status != base::kStatusCodeOk) {
    NNDEPLOY_LOGE("initTensorUsageRecord failed\n");
    return status;
  }

  // 初始化OpBreadth
  status = initOpBreadth();
  if (status != base::kStatusCodeOk) {
    NNDEPLOY_LOGE("initOpBreadth failed\n");
    return status;
  }

  // 记录内存块大小
  size_t total_consumption = 0;
  // 遍历每个张量使用记录
  for (auto& task : op_breadths_) { 
    for (auto& t : task->breadth_) {
        auto it = std::find(ordered_allocated_ids_.begin(), ordered_allocated_ids_.end(), t);
        if (it != ordered_allocated_ids_.end()) continue;
        int prev_offset = 0; // 上一个张量的偏移量
        int best_offset = -1; // 最佳偏移量
        int smallest_gap = INT_MAX; // 最小间隙

        // 遍历已分配的张量，寻找最佳偏移量
        for (const auto& x : ordered_allocated_ids_) {
            int max_first_op = std::max(t->interval_[0], x->interval_[0]);
            int min_last_op = std::min(t->interval_[1], x->interval_[1]);
            // 检查张量是否可以在当前间隙中分配
            if (max_first_op <= min_last_op) {
              int gap = x->offset_ - prev_offset; // 计算间隙大小
              // 如果间隙足够且比当前找到的最小间隙小，则更新最佳偏移量
              if (gap >= t->size_ && gap < smallest_gap) {
                  smallest_gap = gap;
                  best_offset = prev_offset;
              }
              prev_offset = std::max(prev_offset, x->offset_ + static_cast<int>(x->size_));
            }
            // 更新上一个张量的偏移量
            
        }
        if (best_offset==-1) best_offset = prev_offset;
        t->offset_ = best_offset;
        ordered_allocated_ids_.push_back(t);
        total_consumption = std::max(static_cast<int>(total_consumption), best_offset + static_cast<int>(t->size_));
    }
  }
  tensorUsageRecordPrint(tensor_usage_records_);

  NNDEPLOY_LOGE("Total memory size: %zu (OffSetByBreadth)\n", total_consumption);

  return status;
}

base::Status TensorPool1DOffsetCalculateGreedyByBreadth::deallocate() {
  base::Status status = base::kStatusCodeOk;

  for (auto tensor_wrapper : tensor_repository_) {
    auto tensor = tensor_wrapper->tensor_;
    tensor->deallocate();
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

}  // namespace net
}  // namespace nndeploy