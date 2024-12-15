

#include "nndeploy/net/tensor_pool/tensor_pool_1d.h"

#include "nndeploy/net/tensor_pool.h"

namespace nndeploy {
namespace net {
// tensorpool 基类实现
TensorPool1DSharedObject::TensorPool1DSharedObject(
    device::Device *device, std::vector<TensorWrapper *> &tensor_repository,
    std::vector<OpWrapper *> &op_repository)
    : TensorPool(device, tensor_repository, op_repository) {}

TensorPool1DSharedObject::~TensorPool1DSharedObject() {}

base::Status TensorPool1DSharedObject::initTensorUsageRecord() {
  base::Status status = base::kStatusCodeOk;

  for (size_t i = 0; i < tensor_repository_.size(); i++) {
    if (tensor_repository_[i]->is_weight_) {
      continue;
    }
    // if (tensor_repository_[i]->input_output_type_ != kNone) {
    //   NNDEPLOY_LOGI("tensor name = %s.\n",
    //                 tensor_repository_[i]->tensor_->getName().c_str());
    //   if (tensor_repository_[i]->tensor_ != nullptr) {
    //     continue;
    //   }
    // }
    auto tensor_usage_record = std::make_shared<TensorUsageRecord>();
    tensor_usage_record->tensor_wrapper_ = tensor_repository_[i];
    device::TensorDesc tensor_desc = tensor_repository_[i]->tensor_->getDesc();
    device::BufferDesc buffer_desc =
        device_->toBufferDesc(tensor_desc, config_);
    tensor_usage_record->size_ = buffer_desc.getSize();
    int min = op_repository_.size() - 1;
    int max = 0;
    std::vector<int> order_index =
        getOpOrderIndex(tensor_repository_[i]->producers_,
                        tensor_repository_[i]->consumers_, op_repository_);
    for (size_t j = 0; j < order_index.size(); j++) {
      if (order_index[j] < min) {
        min = order_index[j];
      }
      if (order_index[j] > max) {
        max = order_index[j];
      }
    }
    if (tensor_repository_[i]->input_output_type_ != kNone) {
      // 打印tensor_repository_的名字
      NNDEPLOY_LOGE("Tensor name: %s\n",
                    tensor_repository_[i]->tensor_->getName().c_str());
      min = 0;
      max = op_repository_.size() - 1;
    }
    tensor_usage_record->interval_[0] = min;
    tensor_usage_record->interval_[1] = max;
    tensor_usage_records_.push_back(tensor_usage_record);

    // NNDEPLOY_LOGE("tensor name = %s.\n",
    //               tensor_repository_[i]->tensor_->getName().c_str());
    // NNDEPLOY_LOGE("min=%d, max=%d.\n", min, max);
  }
  std::sort(tensor_usage_records_.begin(), tensor_usage_records_.end(),
            [](const std::shared_ptr<TensorUsageRecord> &a,
               const std::shared_ptr<TensorUsageRecord> &b) {
              return a->size_ > b->size_;
            });

  return status;
}

base::Status TensorPool1DSharedObject::deinitTensorUsageRecord() {
  base::Status status = base::kStatusCodeOk;

  tensor_usage_records_.clear();

  return status;
}

base::Status TensorPool1DSharedObject::initOpBreadth() {
  base::Status status = base::kStatusCodeOk;

  for (size_t i = 0; i < op_repository_.size(); i++) {
    auto op_breadth = std::make_shared<OpBreadth>();
    op_breadth->op_wrapper_ = op_repository_[i];
    for (size_t j = 0; j < tensor_usage_records_.size(); j++) {
      if (tensor_usage_records_[j]->interval_[0] <= i &&
          tensor_usage_records_[j]->interval_[1] >= i) {
        op_breadth->breadth_.push_back(tensor_usage_records_[j]);
        op_breadth->size_ += tensor_usage_records_[j]->size_;
      }
    }
    std::sort(op_breadth->breadth_.begin(), op_breadth->breadth_.end(),
              [](const std::shared_ptr<TensorUsageRecord> &a,
                 const std::shared_ptr<TensorUsageRecord> &b) {
                return a->size_ > b->size_;
              });
    op_breadths_.push_back(op_breadth);
  }

  return status;
}
base::Status TensorPool1DSharedObject::deinitOpBreadth() {
  base::Status status = base::kStatusCodeOk;

  op_breadths_.clear();

  return status;
}

base::Status TensorPool1DSharedObject::initPositionalMaximum() {
  base::Status status = base::kStatusCodeOk;

  size_t max = 0;
  for (size_t i = 0; i < op_repository_.size(); i++) {
    max = std::max(max, op_breadths_[i]->breadth_.size());
  }
  positional_maximum_.resize(max, 0);
  for (size_t i = 0; i < max; i++) {
    for (size_t j = 0; j < op_breadths_.size(); j++) {
      if (op_breadths_[j]->breadth_.size() > i) {
        positional_maximum_[i] = std::max(positional_maximum_[i],
                                          op_breadths_[j]->breadth_[i]->size_);
      }
    }
  }

  return status;
}
base::Status TensorPool1DSharedObject::deinitPositionalMaximum() {
  base::Status status = base::kStatusCodeOk;

  positional_maximum_.clear();

  return status;
}
// tensorpool SharedObjectGreedyBySizeImprove 实现
TypeTensorPoolRegister<
    TypeTensorPoolCreator<TensorPool1DSharedObjectGreedyBySizeImprove>>
    g_tensor_pool_1d_shared_object_greedy_by_size_improve_register(
        kTensorPool1DSharedObjectTypeGreedyBySizeImprove);


TensorPool1DSharedObjectGreedyBySizeImprove::
    TensorPool1DSharedObjectGreedyBySizeImprove(
        device::Device *device, std::vector<TensorWrapper *> &tensor_repository,
        std::vector<OpWrapper *> &op_repository)
    : TensorPool1DSharedObject(device, tensor_repository, op_repository) {}

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

  // 按照贪心的办法进行内存分配
  for (size_t i = 0; i < tensor_usage_records_.size(); i++) {
    // 这个都不用
    if (tensor_usage_records_[i]->is_allocated_) {
      continue;
    }

    // NNDEPLOY_LOGE(
    //     "tensor name = %s.\n",
    //     tensor_usage_records_[i]->tensor_wrapper_->tensor_->getName().c_str());

    std::shared_ptr<Chunk> chunk = nullptr;

    // 遍历chunks_
    for (int j = chunks_.size() - 1; j >= 0; j--) {
      size_t chunk_size = chunks_[j]->buffer_->getSize();
      size_t tensor_size = tensor_usage_records_[i]->size_;
      if (chunk_size >= tensor_size) {
        // for (const auto &interval : chunks_[j]->intervals_) {
        //   NNDEPLOY_LOGE("Chunk interval: [%d, %d]\n", interval[0],
        //   interval[1]);
        // }
        // NNDEPLOY_LOGE("TensorUsageRecord interval: [%d, %d]\n",
        //               tensor_usage_records_[i]->interval_[0],
        //               tensor_usage_records_[i]->interval_[1]);
        bool flag = isInterval(tensor_usage_records_[i]->interval_,
                               chunks_[j]->intervals_);
        if (!flag) {
          // NNDEPLOY_LOGE(
          //     "Tensor name: %s\n",
          //     tensor_usage_records_[i]->tensor_wrapper_->name_.c_str());
          // NNDEPLOY_LOGE("TensorUsageRecord interval: [%d, %d]\n",
          //               tensor_usage_records_[i]->interval_[0],
          //               tensor_usage_records_[i]->interval_[1]);
          chunk = chunks_[j];
          chunks_[j]->intervals_.push_back(tensor_usage_records_[i]->interval_);
          break;
        }
      }
    }

    // 不存在
    if (chunk == nullptr) {
      chunk = std::make_shared<Chunk>();
      // NNDEPLOY_LOGE("tensor name = %s.\n",
      //               tensor_usage_records_[i]
      //                   ->tensor_wrapper_->tensor_->getName()
      //                   .c_str());
      // NNDEPLOY_LOGE("size_=%ld.\n", tensor_usage_records_[i]->size_);
      // tensor_usage_records_[i]->tensor_wrapper_->tensor_->getDesc().print();
      size_t size = tensor_usage_records_[i]->size_;
      chunk->buffer_ = new device::Buffer(device_, size);
      if (chunk->buffer_ == nullptr) {
        NNDEPLOY_LOGE("chunk->buffer_ is empty\n");
        return base::kStatusCodeErrorNullParam;
      }
      std::array<int, 2> interval = tensor_usage_records_[i]->interval_;
      chunk->intervals_.push_back(interval);
      chunks_.push_back(chunk);
    }

    // 与tensor关联
    tensor_usage_records_[i]->is_allocated_ = true;
    device::Buffer *buffer = new device::Buffer(*chunk->buffer_);
    device::TensorDesc tensor_desc =
        tensor_usage_records_[i]->tensor_wrapper_->tensor_->getDesc();
    device::BufferDesc buffer_desc =
        device_->toBufferDesc(tensor_desc, base::IntVector());
    if (!buffer->justModify(buffer_desc)) {
      NNDEPLOY_LOGE("tensor name = %s.\n",
                    tensor_usage_records_[i]->tensor_wrapper_->name_.c_str());
      NNDEPLOY_LOGE("buffer->justModify failed\n");
      return base::kStatusCodeErrorInvalidValue;
    }
    tensor_usage_records_[i]->tensor_wrapper_->tensor_->justModify(buffer,
                                                                   false);
  }

  // 统计tensor的个数，并累加大小
  // size_t total_tensor_count = 0;
  // size_t total_memory_size = 0;
  // for (const auto &tensor_usage_record : tensor_usage_records_) {
  //   total_tensor_count++;
  //   total_memory_size += tensor_usage_record->size_;
  // }
  // NNDEPLOY_LOGE("Total tensor count: %zu\n", total_tensor_count);
  // NNDEPLOY_LOGE("Total memory size: %zu\n", total_memory_size);
  tensorUsageRecordPrint(tensor_usage_records_);

  // 统计chunk的个数，并累加大小
  // size_t total_chunk_count = 0;
  // size_t total_chunk_size = 0;
  // for (const auto &chunk : chunks_) {
  //   total_chunk_count++;
  //   total_chunk_size += chunk->buffer_->getSize();
  // }
  // NNDEPLOY_LOGE("Total chunk count: %zu\n", total_chunk_count);
  // NNDEPLOY_LOGE("Total chunk size: %zu\n", total_chunk_size);
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

// tensorpool SharedObjectGreedyByBreadth 实现

TypeTensorPoolRegister<
    TypeTensorPoolCreator<TensorPool1DSharedObjectGreedyByBreadth>>
    g_tensor_pool_1d_shared_object_greedy_by_breadth_register(
        kTensorPool1DSharedObjectTypeGreedyByBreadth);

TensorPool1DSharedObjectGreedyByBreadth::
    TensorPool1DSharedObjectGreedyByBreadth(
        device::Device *device, std::vector<TensorWrapper *> &tensor_repository,
        std::vector<OpWrapper *> &op_repository)
    : TensorPool1DSharedObject(device, tensor_repository, op_repository) {}

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

// tensorpool OffsetCalculateGreedyBySize 实现

TypeTensorPoolRegister<
    TypeTensorPoolCreator<TensorPool1DOffsetCalculateGreedyBySize>>
    g_tensor_pool_1d_offset_calculate_greedy_by_size_register(
        kTensorPool1DOffsetCalculateTypeGreedyBySize);


TensorPool1DOffsetCalculateGreedyBySize::
    TensorPool1DOffsetCalculateGreedyBySize(
        device::Device *device, std::vector<TensorWrapper *> &tensor_repository,
        std::vector<OpWrapper *> &op_repository)
    : TensorPool1DSharedObject(device, tensor_repository, op_repository) {}

TensorPool1DOffsetCalculateGreedyBySize::
    ~TensorPool1DOffsetCalculateGreedyBySize() {}

base::Status TensorPool1DOffsetCalculateGreedyBySize::allocate() {
  base::Status status = base::kStatusCodeOk;

  // 初始化TensorUsageRecord, 对tensor大小进行排序
  status = initTensorUsageRecord();
  if (status != base::kStatusCodeOk) {
    NNDEPLOY_LOGE("initTensorUsageRecord failed\n");
    return status;
  }

  // 记录内存块大小
  size_t totalConsumption = 0;
  // 遍历每个张量使用记录
  for (auto& t : tensor_usage_records_) {
      int prevOffset = 0; // 上一个张量的偏移量
      int bestOffset = -1; // 最佳偏移量
      int smallestGap = INT_MAX; // 最小间隙

      // 遍历已分配的张量，寻找最佳偏移量
      for (const auto& x : ordered_allocated_ids_) {
          int maxFirstOp = std::max(t->interval_[0], x->interval_[0]);
          int minLastOp = std::min(t->interval_[1], x->interval_[1]);
          // 检查张量是否可以在当前间隙中分配
          if (maxFirstOp <= minLastOp) {
              int gap = x->offset - prevOffset; // 计算间隙大小
              // 如果间隙足够且比当前找到的最小间隙小，则更新最佳偏移量
              if (gap >= t->size_ && gap < smallestGap) {
                  smallestGap = gap;
                  bestOffset = prevOffset;
              }
              prevOffset = std::max(prevOffset, x->offset + static_cast<int>(x->size_));

          }
          // 更新上一个张量的偏移量
          
      }
      // NNDEPLOY_LOGE("bestoffset: %d \n", bestOffset);
      if (bestOffset==-1) bestOffset = prevOffset;
      t->offset = bestOffset;
      NNDEPLOY_LOGE("bestoffset: %d \n", bestOffset);
      ordered_allocated_ids_.push_back(t);
      NNDEPLOY_LOGE("firstop: %d \n", t->interval_[0]);
      NNDEPLOY_LOGE("lastop: %d \n", t->interval_[1]);
      NNDEPLOY_LOGE("size: %d \n", t->size_);
      // NNDEPLOY_LOGE("order size: %d \n", ordered_allocated_ids_.size());
      totalConsumption = std::max(static_cast<int>(totalConsumption), bestOffset + static_cast<int>(t->size_));
  }

  tensorUsageRecordPrint(tensor_usage_records_);

  NNDEPLOY_LOGE("Total memory size: %zu (OffSet)\n", totalConsumption);

  return status;
}
base::Status TensorPool1DOffsetCalculateGreedyBySize::deallocate() {
  base::Status status = base::kStatusCodeOk;

  for (auto tensor_wrapper : tensor_repository_) {
    auto tensor = tensor_wrapper->tensor_;
    tensor->deallocate();
  }

  // status = deinitOpBreadth();
  // if (status != base::kStatusCodeOk) {
  //   NNDEPLOY_LOGE("deinitOpBreadth failed\n");
  //   return status;
  // }

  status = deinitTensorUsageRecord();
  if (status != base::kStatusCodeOk) {
    NNDEPLOY_LOGE("deinitTensorUsageRecord failed\n");
    return status;
  }

  return status;
}

// tensorpool OffsetCalculateGreedyByBreadth 实现

TypeTensorPoolRegister<
    TypeTensorPoolCreator<TensorPool1DOffsetCalculateGreedyByBreadth>>
    g_tensor_pool_1d_offset_calculate_greedy_by_breadth_register(
        kTensorPool1DOffsetCalculateTypeGreedyByBreadth);


TensorPool1DOffsetCalculateGreedyByBreadth::
    TensorPool1DOffsetCalculateGreedyByBreadth(
        device::Device *device, std::vector<TensorWrapper *> &tensor_repository,
        std::vector<OpWrapper *> &op_repository)
    : TensorPool1DSharedObject(device, tensor_repository, op_repository) {}

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
        int prevOffset = 0; // 上一个张量的偏移量
        int bestOffset = -1; // 最佳偏移量
        int smallestGap = INT_MAX; // 最小间隙

        // 遍历已分配的张量，寻找最佳偏移量
        for (const auto& x : ordered_allocated_ids_) {
            int maxFirstOp = std::max(t->interval_[0], x->interval_[0]);
            int minLastOp = std::min(t->interval_[1], x->interval_[1]);
            // 检查张量是否可以在当前间隙中分配
            if (maxFirstOp <= minLastOp) {
              int gap = x->offset - prevOffset; // 计算间隙大小
              // 如果间隙足够且比当前找到的最小间隙小，则更新最佳偏移量
              if (gap >= t->size_ && gap < smallestGap) {
                  smallestGap = gap;
                  bestOffset = prevOffset;
              }
              prevOffset = std::max(prevOffset, x->offset + static_cast<int>(x->size_));
            }
            // 更新上一个张量的偏移量
            
        }
        if (bestOffset==-1) bestOffset = prevOffset;
        t->offset = bestOffset;
        ordered_allocated_ids_.push_back(t);
        total_consumption = std::max(static_cast<int>(total_consumption), bestOffset + static_cast<int>(t->size_));
    }
  }
  tensorUsageRecordPrint(tensor_usage_records_);

  NNDEPLOY_LOGE("Total memory size: %zu (OffSet)\n", total_consumption);

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
