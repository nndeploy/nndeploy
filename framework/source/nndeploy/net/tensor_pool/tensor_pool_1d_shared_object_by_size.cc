#include "nndeploy/net/tensor_pool/tensor_pool_1d_shared_object_by_size.h"

#include "nndeploy/net/tensor_pool.h"

namespace nndeploy {
namespace net {

// tensorpool SharedObjectGreedyBySize 实现
TypeTensorPoolRegister<
    TypeTensorPoolCreator<TensorPool1DSharedObjectGreedyBySize>>
    g_tensor_pool_1d_shared_object_greedy_by_size_register(
        kTensorPool1DSharedObjectTypeGreedyBySize);

TensorPool1DSharedObjectGreedyBySize::TensorPool1DSharedObjectGreedyBySize(
    device::Device *device, std::vector<TensorWrapper *> &tensor_repository,
    std::vector<OpWrapper *> &op_repository)
    : TensorPool1D(device, tensor_repository, op_repository) {}

TensorPool1DSharedObjectGreedyBySize::~TensorPool1DSharedObjectGreedyBySize() {}

base::Status TensorPool1DSharedObjectGreedyBySize::allocate() {
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

  tensorUsageRecordPrint(tensor_usage_records_);
  chunkPrint(chunks_);

  return status;
}
base::Status TensorPool1DSharedObjectGreedyBySize::deallocate() {
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