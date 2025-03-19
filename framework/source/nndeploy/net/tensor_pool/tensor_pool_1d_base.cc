#include "nndeploy/net/tensor_pool/tensor_pool_1d_base.h"

#include "nndeploy/net/tensor_pool.h"

namespace nndeploy {
namespace net {

// tensorpool 基类实现
TensorPool1D::TensorPool1D(device::Device *device,
                           std::vector<TensorWrapper *> &tensor_repository,
                           std::vector<OpWrapper *> &op_repository)
    : TensorPool(device, tensor_repository, op_repository) {}

TensorPool1D::~TensorPool1D() {}

base::Status TensorPool1D::initTensorUsageRecord() {
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

base::Status TensorPool1D::deinitTensorUsageRecord() {
  base::Status status = base::kStatusCodeOk;

  tensor_usage_records_.clear();

  return status;
}

base::Status TensorPool1D::initOpBreadth() {
  base::Status status = base::kStatusCodeOk;

  int max_breadth_length = 0;
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

    // NNDEPLOY_LOGE("op_breadth[%s, %d] = %ld, length = %ld.\n",
    //               op_breadth->op_wrapper_->op_->getName().c_str(), i,
    //               op_breadth->size_, op_breadth->breadth_.size());
    // for (size_t j = 0; j < op_breadth->breadth_.size(); j++) {
    //   NNDEPLOY_LOGE("tensor_usage_record[%s, %d] = %ld.\n",
    //                 op_breadth->breadth_[j]->tensor_wrapper_->tensor_->getName().c_str(),
    //                 i, op_breadth->breadth_[j]->size_);
    // }
    max_breadth_length =
        std::max(max_breadth_length, static_cast<int>(op_breadth->breadth_.size()));
  }

  std::sort(op_breadths_.begin(), op_breadths_.end(),
            [](const std::shared_ptr<OpBreadth> &a,
               const std::shared_ptr<OpBreadth> &b) {
              return a->size_ > b->size_;
            });
  // NNDEPLOY_LOGE("max_breadth_length = %d.\n", max_breadth_length);

  return status;
}
base::Status TensorPool1D::deinitOpBreadth() {
  base::Status status = base::kStatusCodeOk;

  op_breadths_.clear();

  return status;
}

base::Status TensorPool1D::initPositionalMaximum() {
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
base::Status TensorPool1D::deinitPositionalMaximum() {
  base::Status status = base::kStatusCodeOk;

  positional_maximum_.clear();

  return status;
}

}  // namespace net
}  // namespace nndeploy