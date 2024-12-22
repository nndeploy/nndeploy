#include "nndeploy/net/tensor_pool/tensor_pool_shared_object_by_breadth.h"

#include "nndeploy/net/tensor_pool.h"

namespace nndeploy {
namespace net {

// tensorpool SharedObjectGreedyByBreadth 实现

TypeTensorPoolRegister<
    TypeTensorPoolCreator<TensorPool1DSharedObjectGreedyByBreadth>>
    g_tensor_pool_1d_shared_object_greedy_by_breadth_register(
        kTensorPool1DSharedObjectTypeGreedyByBreadth);

TensorPool1DSharedObjectGreedyByBreadth::
    TensorPool1DSharedObjectGreedyByBreadth(
        device::Device *device, std::vector<TensorWrapper *> &tensor_repository,
        std::vector<OpWrapper *> &op_repository)
    : TensorPool1D(device, tensor_repository, op_repository) {}

TensorPool1DSharedObjectGreedyByBreadth::
    ~TensorPool1DSharedObjectGreedyByBreadth() {}

base::Status TensorPool1DSharedObjectGreedyByBreadth::allocate() {
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

  for (const auto &task : op_breadths_) {
    // 遍历当前task的所有tensor usuage
    for (auto &tensor_usuage : task->breadth_) {
      if (assigned_tensors_.count(tensor_usuage) != 0) {
        continue;
      }

      NNDEPLOY_LOGE("tensor name = %s.\n",
                    tensor_usuage->tensor_wrapper_->tensor_->getName().c_str());

      std::shared_ptr<Chunk> best_chunk = nullptr;  // 最佳适配的chunk

      for (int i = 0; i < chunks_.size(); i++) {
        if (best_chunk != nullptr) {
          const size_t best_size = chunk_sizes_[best_chunk];
          const size_t cur_size = chunk_sizes_[chunks_[i]];
          const size_t tensor_size = tensor_usuage->size_;
          if (best_size < tensor_size) {
            if (cur_size <=
                best_size) {  // 最佳的块大小小于当前tensor,
                              // 但是当前块比当前tensor还小,那么当前块就不能成为最佳块（会扩容更多）
              continue;
            }
          } else if (cur_size < tensor_size || cur_size >= best_size) {
            // 1.当前块的大小比当前tensor小，那么当前块需要扩充，不适合
            // 2.当前块比最佳块要大，那么可能浪费空间，不适合
            continue;
          }
        }

        // 通过查看当前chunk已被分配给哪些tensor，来决定是否能分配给当前tensor
        // 如果没有发现时间区间的交集，当前Chunk可能是合适的候选对象，考虑将其作为最佳Chunk。
        if (isInterval(tensor_usuage->interval_, chunks_[i]->intervals_)) {
          continue;
        }
        best_chunk = chunks_[i];
      }
      if (best_chunk ==
          nullptr) {  // 需要创建一个新的，此时延迟开辟，仅记录该Chunk的存在和size
        NNDEPLOY_LOGE("Create a new chunk\n");
        best_chunk = std::make_shared<Chunk>();
        NNDEPLOY_LOGE(
            "tensor name = %s.\n",
            tensor_usuage->tensor_wrapper_->tensor_->getName().c_str());
        NNDEPLOY_LOGE("size_=%ld.\n", tensor_usuage->size_);
        size_t size = tensor_usuage->size_;
        chunk_sizes_[best_chunk] = size;
        std::array<int, 2> interval = tensor_usuage->interval_;
        best_chunk->intervals_.push_back(interval);
        chunks_.push_back(best_chunk);

      } else {
        chunk_sizes_[best_chunk] =
            std::max(chunk_sizes_[best_chunk], tensor_usuage->size_);  // 扩容
      }
      chunk_schedules_[best_chunk].insert(tensor_usuage);
      assigned_tensors_.insert(tensor_usuage);
    }
  }

  // 开辟内存
  for (auto kv : chunk_schedules_) {
    auto chunk = kv.first;
    auto tensors = kv.second;
    chunk->buffer_ = new device::Buffer(device_, chunk_sizes_[chunk]);
    if (chunk->buffer_ == nullptr) {
      NNDEPLOY_LOGE("chunk->buffer_ alloc failed\n");
      return base::kStatusCodeErrorNullParam;
    }

    // 与tensor关联
    for (auto tensor : tensors) {
      tensor->is_allocated_ = true;
      tensor->tensor_wrapper_->tensor_->justModify(chunk->buffer_);
    }
  }

  return status;
}
base::Status TensorPool1DSharedObjectGreedyByBreadth::deallocate() {
  base::Status status = base::kStatusCodeOk;

  for (auto tensor_wrapper : tensor_repository_) {
    auto tensor = tensor_wrapper->tensor_;
    tensor->deallocate();
  }

  for (size_t i = 0; i < chunks_.size(); i++) {
    if (chunks_[i]->buffer_ != nullptr) {
      delete chunks_[i]->buffer_;
    }
  }

  status = deinitPositionalMaximum();
  if (status != base::kStatusCodeOk) {
    NNDEPLOY_LOGE("deinitPositionalMaximum failed\n");
    return status;
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

  chunk_sizes_.clear();
  chunk_schedules_.clear();

  return status;
}


}  // namespace net
}  // namespace nndeploy