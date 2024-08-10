

#include "nndeploy/net/tensor_pool_1d.h"

namespace nndeploy {
namespace net {

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
    auto tensor_usage_record = std::make_shared<TensorUsageRecord>();
    tensor_usage_record->tensor_wrapper_ = tensor_repository_[i];
    device::TensorDesc tensor_desc = tensor_repository_[i]->tensor_->getDesc();
    device::BufferDesc buffer_desc =
        device_->toBufferDesc(tensor_desc, config_);
    tensor_usage_record->size_ = buffer_desc.size_[0];
    int min = op_repository.size() - 1;
    int max = 0;
    std::vector<int> order_index =
        getOpOrderIndex(tensor_repository_[i]->producers_,
                        tensor_repository_[i]->consumers_, op_repository);
    for (size_t j = 0; j < order_index.size(); j++) {
      if (order_index[j] < min) {
        min = order_index[j];
      }
      if (order_index[j] > max) {
        max = order_index[j];
      }
    }
    tensor_usage_record->interval_[0] = min;
    tensor_usage_record->interval_[1] = max;
    tensor_usage_records_.push_back(tensor_usage_record);
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

  for (size_t i = 0; i < op_repository.size(); i++) {
    auto op_breadth = std::make_shared<OpBreadth>();
    op_breadth->op_wrapper_ = op_repository[i];
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
  for (size_t i = 0; i < op_repository.size(); i++) {
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

kTensorPool1DSharedObjectTypeGreedyBySizeImprove::
    kTensorPool1DSharedObjectTypeGreedyBySizeImprove(
        device::Device *device, std::vector<TensorWrapper *> &tensor_repository,
        std::vector<OpWrapper *> &op_repository)
    : TensorPool1DSharedObject(device, tensor_repository, op_repository) {}

kTensorPool1DSharedObjectTypeGreedyBySizeImprove::
    ~kTensorPool1DSharedObjectTypeGreedyBySizeImprove() {}

base::Status kTensorPool1DSharedObjectTypeGreedyBySizeImprove::allocate() {
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

    std::shared_ptr<Chunk> chunk = nullptr;
    // 遍历chunks_
    for (size_t j = chunks_.size() - 1; j >= 0; j--) {
      size_t chunk_size = chunks_[j]->buffer_->getSize();
      size_t tensor_size = tensor_usage_records_[i]->size_;
      if (chunk_size >= tensor_size) {
        bool flag = isInterval(tensor_usage_records_[i]->interval_,
                               chunks_[j]->intervals_);
        if (!flag) {
          chunk = chunks_[j];
          chunks_[j]->intervals_.push_back(tensor_usage_records_[i]->interval_);
        }
        break;
      }
    }
    // 不存在
    if (chunk == nullptr) {
      chunk = std::make_shared<Chunk>();
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
    tensor_usage_records_[i]->tensor_wrapper_->tensor_->justModify(
        chunk->buffer_);
  }

  return status;
}
base::Status kTensorPool1DSharedObjectTypeGreedyBySizeImprove::deallocate() {
  base::Status status = base::kStatusCodeOk;

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
