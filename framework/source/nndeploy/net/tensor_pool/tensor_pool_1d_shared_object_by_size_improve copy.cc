#include "nndeploy/net/tensor_pool/tensor_pool_1d_shared_object_by_size_improve.h"

#include "nndeploy/net/tensor_pool.h"

namespace nndeploy {
namespace net {

// tensorpool SharedObjectGreedyBySizeImprove 实现
TypeTensorPoolRegister<
    TypeTensorPoolCreator<TensorPool1DSharedObjectGreedyBySizeImprove>>
    g_tensor_pool_1d_shared_object_greedy_by_size_improve_register(
        kTensorPool1DSharedObjectTypeGreedyBySizeImprove);


TensorPool1DSharedObjectGreedyBySizeImprove::
    TensorPool1DSharedObjectGreedyBySizeImprove(
        device::Device *device, std::vector<TensorWrapper *> &tensor_repository,
        std::vector<OpWrapper *> &op_repository)
    : TensorPool1D(device, tensor_repository, op_repository) {}

TensorPool1DSharedObjectGreedyBySizeImprove::
    ~TensorPool1DSharedObjectGreedyBySizeImprove() {}

base::Status TensorPool1DSharedObjectGreedyBySizeImprove::allocate() {
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

  // 初始化PositionalMaximum
  status = initPositionalMaximum();
  if (status != base::kStatusCodeOk) {
    NNDEPLOY_LOGE("initPositionalMaximum failed\n");
    return status;
  }

  // 按位置最大值分阶段进行分配
  for (size_t pos_max_idx = 0; pos_max_idx < positional_maximum_.size(); ++pos_max_idx) {
    size_t current_max = positional_maximum_[pos_max_idx];
    std::vector<std::shared_ptr<TensorUsageRecord>> current_stage_tensors;

    // 收集当前阶段的张量
    for (auto& tensor_record : tensor_usage_records_) {
      if (!tensor_record->is_allocated_ && tensor_record->size_ <= current_max) {
        current_stage_tensors.push_back(tensor_record);
      }
    }

    // 对当前阶段的张量按使用间隔进行排序
    std::sort(current_stage_tensors.begin(), current_stage_tensors.end(),
              [](const std::shared_ptr<TensorUsageRecord>& a, const std::shared_ptr<TensorUsageRecord>& b) {
                return a->interval_[0] < b->interval_[0];
              });

    // 分配当前阶段的张量
    for (auto& tensor_record : current_stage_tensors) {
      std::shared_ptr<Chunk> best_chunk = nullptr;
      int min_interval_distance = INT_MAX;

      // 寻找最佳共享对象
      for (auto& chunk : chunks_) {
        if (chunk->buffer_->getSize() >= tensor_record->size_) {
          bool flag = isInterval(tensor_record->interval_, chunk->intervals_);
          if (!flag) {
            int last_interval = chunk->intervals_.empty() ? 0 : chunk->intervals_.back()[1];
            int interval_distance = std::min(abs(tensor_record->interval_[0] - last_interval),
                                              abs(tensor_record->interval_[1] - last_interval));
            if (interval_distance < min_interval_distance) {
              best_chunk = chunk;
              min_interval_distance = interval_distance;
            }
          }
        }
      }

      // 如果没有找到合适的共享对象，则创建一个新的
      if (best_chunk == nullptr) {
        best_chunk = std::make_shared<Chunk>();
        size_t size = tensor_record->size_;
        best_chunk->buffer_ = new device::Buffer(device_, size);
        if (best_chunk->buffer_ == nullptr) {
          NNDEPLOY_LOGE("best_chunk->buffer_ is empty\n");
          return base::kStatusCodeErrorNullParam;
        }
        chunks_.push_back(best_chunk);
      }

      // 分配张量到共享对象
      best_chunk->intervals_.push_back(tensor_record->interval_);
      tensor_record->is_allocated_ = true;
      device::Buffer *buffer = new device::Buffer(*best_chunk->buffer_);
      device::TensorDesc tensor_desc = tensor_record->tensor_wrapper_->tensor_->getDesc();
      device::BufferDesc buffer_desc = device_->toBufferDesc(tensor_desc, base::IntVector());
      if (!buffer->justModify(buffer_desc)) {
        NNDEPLOY_LOGE("tensor name = %s.\n", tensor_record->tensor_wrapper_->name_.c_str());
        NNDEPLOY_LOGE("buffer->justModify failed\n");
        return base::kStatusCodeErrorInvalidValue;
      }
      tensor_record->tensor_wrapper_->tensor_->justModify(buffer, false);
    }
  }


  tensorUsageRecordPrint(tensor_usage_records_);

  chunkPrint(chunks_);

  return status;
}

base::Status TensorPool1DSharedObjectGreedyBySizeImprove::deallocate() {
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

  return status;
}


}  // namespace net
}  // namespace nndeploy