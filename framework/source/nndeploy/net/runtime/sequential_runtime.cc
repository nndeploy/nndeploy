

#include "nndeploy/net/runtime/sequential_runtime.h"

#include "nndeploy/base/time_profiler.h"

namespace nndeploy {
namespace net {

TypeRuntimeRegister<TypeRuntimeCreator<SequentialRuntime>>
    g_sequential_runtime_register_seq(
        base::ParallelType::kParallelTypeSequential);

TypeRuntimeRegister<TypeRuntimeCreator<SequentialRuntime>>
    g_sequential_runtime_register_none(base::ParallelType::kParallelTypeNone);

SequentialRuntime::SequentialRuntime(const base::DeviceType& device_type)
    : Runtime(device_type) {};
SequentialRuntime::~SequentialRuntime() {};

base::Status SequentialRuntime::init(
    std::vector<TensorWrapper*>& tensor_repository,
    std::vector<OpWrapper*>& op_repository,
    std::vector<device::Tensor*>& input_tensors,
    std::vector<device::Tensor*>& output_tensors, bool is_dynamic_shape,
    base::ShapeMap max_shape, TensorPoolType tensor_pool_type,
    bool is_external_tensor_pool_memory) {
  base::Status status = base::kStatusCodeOk;
  device::Device* device = device::getDevice(device_type_);

  if (tensor_pool_) return base::kStatusCodeErrorInvalidValue;
  tensor_repository_ = tensor_repository;
  op_repository_ = op_repository;
  is_dynamic_shape_ = is_dynamic_shape;
  max_shape_ = max_shape;
  // 输入输出tensor
  input_tensors_ = input_tensors;
  output_tensors_ = output_tensors;

  for (auto iter : op_repository) {
    iter->op_->setDeviceType(device_type_);
  }
  // 默认流
  if (!is_external_stream_ && stream_ == nullptr) {
    stream_ = device::createStream(device_type_);
  }
  for (auto iter : op_repository) {
    iter->op_->setStream(stream_);
  }

  // # 激活值的tensor分配
  tensor_pool_type_ = tensor_pool_type;
  tensor_pool_ = createTensorPool(tensor_pool_type_, device, tensor_repository,
                                  op_repository);
  is_external_tensor_pool_memory_ = is_external_tensor_pool_memory;
  if (!tensor_pool_) return base::kStatusCodeErrorNotSupport;
  tensor_pool_->setIsExternal(is_external_tensor_pool_memory);
  /**
   * @brief
   * 如果是动态shape且max_shape为空时，那么不需要分配tensor
   */
  bool flag = is_dynamic_shape && max_shape.empty();
  if (!flag) {
    status = tensor_pool_->allocate();
    if (status != base::kStatusCodeOk) {
      NNDEPLOY_LOGE("tensor_pool_ allocate failed\n");
      return status;
    }
    is_pure_dynamic_shape_ = false;
  } else {
    status = tensor_pool_->initTensorUsageRecordMap();
    if (status != base::kStatusCodeOk) {
      NNDEPLOY_LOGE("tensor_pool_ initTensorUsageRecordMap failed\n");
      return status;
    }
    status = tensor_pool_->initOpIndexMap();
    if (status != base::kStatusCodeOk) {
      NNDEPLOY_LOGE("tensor_pool_ initOpIndexMap failed\n");
      return status;
    }
    is_pure_dynamic_shape_ = true;
  }

  // # op的初始化
  // ## 权重转换
  for (auto iter : op_repository) {
    iter->op_->setInitializedFlag(false);
    status = iter->op_->init();
    if (status != base::kStatusCodeOk) {
      NNDEPLOY_LOGE("Node %s init failed\n", iter->op_->getName().c_str());
      return status;
    }
    iter->op_->setInitializedFlag(true);
  }

  tensor_repository_ = tensor_repository;
  op_repository_ = op_repository;
  is_dynamic_shape_ = is_dynamic_shape;
  max_shape_ = max_shape;
  return status;
}
base::Status SequentialRuntime::deinit() {
  base::Status result = base::kStatusCodeOk;
  if (!workspace_is_external_ && workspace_) device::getDevice(device_type_)->deallocate(workspace_);
  workspace_ = nullptr; workspace_size_ = 0;
  for (auto *entry : op_repository_) {
    if (!entry->op_->getInitialized()) continue;
    auto status = entry->op_->deinit();
    if (result == base::kStatusCodeOk) result = status;
    entry->op_->setInitializedFlag(false);
    entry->op_->setWorkspace(nullptr);
  }
  if (tensor_pool_) {
    auto status = tensor_pool_->deallocate();
    if (result == base::kStatusCodeOk) result = status;
    delete tensor_pool_; tensor_pool_ = nullptr;
  }
  op_repository_.clear(); tensor_repository_.clear();
  return result;
}

// 可以性能优化
base::Status SequentialRuntime::reshape(base::ShapeMap& shape_map) {
  base::Status status = base::kStatusCodeOk;
  if (!is_dynamic_shape_) {
    NNDEPLOY_LOGE("reshape is not supported in static shape\n");
    return base::kStatusCodeErrorInvalidParam;
  }
  // Validate every requested shape before releasing the current allocation.
  for (const auto &entry : shape_map) {
    auto *wrapper = findTensorWrapper(tensor_repository_, entry.first);
    if (!wrapper) return base::kStatusCodeErrorInvalidParam;
    device::Tensor descriptor(wrapper->tensor_->getDesc(), entry.first);
    auto status = descriptor.reshape(entry.second);
    if (status != base::kStatusCodeOk) return status;
  }
  bool change_flag = false;
  bool is_reallocate = false;
  for (auto iter : shape_map) {
    std::string name = iter.first;
    base::IntVector shape = iter.second;
    for (auto tensor_wrapper : tensor_repository_) {
      auto tensor = tensor_wrapper->tensor_;
      if (tensor->getName() == name) {
        base::IntVector old_shape = tensor->getShape();
        if (base::shapeEqual(old_shape, shape, 0, -1)) {
          continue;
        }
        change_flag = true;
        if (!max_shape_.empty()) {
          if (max_shape_.find(name) != max_shape_.end()) {
            base::IntVector max_shape = max_shape_[name];
            if (shape.size() != max_shape.size()) is_reallocate = true;
            for (size_t i = 0; !is_reallocate && i < shape.size(); ++i) {
              if (shape[i] > max_shape[i]) {
                is_reallocate = true;
                break;
              }
            }
          }
        } else {
          is_reallocate = true;
        }
      } else {
        is_reallocate = true;
      }
    }
  }
  if (change_flag) {
    if (is_reallocate) {
      status = tensor_pool_->deallocate();
      if (status != base::kStatusCodeOk) {
        NNDEPLOY_LOGE("tensor_pool_ allocate failed\n");
        return status;
      }
    }
    for (auto iter : shape_map) {
      std::string name = iter.first;
      base::IntVector shape = iter.second;
      for (auto tensor_wrapper : tensor_repository_) {
        auto tensor = tensor_wrapper->tensor_;
        if (tensor->getName() == name) {
          status = tensor->reshape(shape);
          if (status != base::kStatusCodeOk) return status;
        }
      }
    }
    for (auto iter : op_repository_) {
      status = iter->op_->inferShape();
      if (status != base::kStatusCodeOk) {
        NNDEPLOY_LOGE("Node %s inferShape failed\n",
                      iter->op_->getName().c_str());
        is_pure_dynamic_shape_ = true;
        return base::kStatusCodeOk;
      } else {
        is_pure_dynamic_shape_ = false;
      }
    }
    if (is_reallocate) {
      NNDEPLOY_LOGI("tensor_pool_ reallocate\n");
      status = tensor_pool_->allocate();
      if (status != base::kStatusCodeOk) {
        NNDEPLOY_LOGE("tensor_pool_ allocate failed\n");
        return status;
      }
    }
  }
  return status;
}

base::Status SequentialRuntime::preRun() {
  base::Status status = base::kStatusCodeOk;
  for (auto iter : op_repository_) {
    status = iter->op_->preRun();
    if (status != base::kStatusCodeOk) {
      NNDEPLOY_LOGE("Node %s preRun failed\n", iter->op_->getName().c_str());
      return status;
    }
  }
  // NNDEPLOY_LOGI("preRun ok!\n");
  return status;
}
base::Status SequentialRuntime::run() {
  base::Status status = base::kStatusCodeOk;
  device::Device* device = device::getDevice(device_type_);
  // workspace
  if (workspace_size_ == 0 || is_dynamic_shape_) {
    if (workspace_ != nullptr) {
      device->deallocate(workspace_);
      workspace_ = nullptr;
    }
    workspace_size_ = 0;
    for (auto iter : op_repository_) {
      uint64_t workspace_size = iter->op_->getWorkspaceSize();
      if (workspace_size > workspace_size_) {
        workspace_size_ = workspace_size;
      }
    }
    if (workspace_size_ > 0) {
      workspace_ = device->allocate(workspace_size_);
      if (!workspace_) return base::kStatusCodeErrorOutOfMemory;
    } else {
      workspace_size_ = 0;  // No allocation is needed.
    }
    for (auto iter : op_repository_) {
      iter->op_->setWorkspace(workspace_);
    }
  }
  // 运行
  NNDEPLOY_TIME_POINT_START("net->run()");
  for (auto iter : op_repository_) {
    if (is_pure_dynamic_shape_) {
      status = iter->op_->inferShape();
      if (status != base::kStatusCodeOk) {
        NNDEPLOY_LOGE("Node %s inferShape failed\n",
                      iter->op_->getName().c_str());
        return status;
      }
      status = tensor_pool_->allocateOp(iter->op_);
      if (status != base::kStatusCodeOk) {
        NNDEPLOY_LOGE("tensor_pool_ allocate node[%s] failed\n",
                      iter->op_->getName().c_str());
        return status;
      }
    }
    // NNDEPLOY_LOGE("Node %s run\n", iter->op_->getName().c_str());
    status = iter->op_->run();
    if (status != base::kStatusCodeOk) {
      NNDEPLOY_LOGE("Node %s run failed\n", iter->op_->getName().c_str());
      return status;
    }
    if (is_pure_dynamic_shape_) {
      status = tensor_pool_->deallocateOp(iter->op_);
      if (status != base::kStatusCodeOk) {
        NNDEPLOY_LOGE("tensor_pool_ deallocate node[%s] failed\n",
                      iter->op_->getName().c_str());
        return status;
      }
    }
  }
  NNDEPLOY_TIME_POINT_END("net->run()");
  // NNDEPLOY_LOGI("run ok!\n");
#if 0
  NNDEPLOY_TIME_POINT_START("stream_->synchronize()");
  status = stream_->synchronize();
  if (status != base::kStatusCodeOk) {
    NNDEPLOY_LOGE("stream_->synchronize() failed\n");
    return status;
  }
  NNDEPLOY_TIME_POINT_END("stream_->synchronize()");
  for (auto tensor : tensor_repository_) {
    if (tensor->tensor_->getName() == "images") {
      std::string filename =
          "sequential_input_" + tensor->tensor_->getName() + ".csv";
      std::ofstream file_stream(filename.c_str());
      tensor->tensor_->print(file_stream);
      file_stream.close();
    }
  }
  for (auto tensor : tensor_repository_) {
    if (tensor->tensor_->getName() == "output0") {
      std::string filename =
          "sequential_output_" + tensor->tensor_->getName() + ".csv";
      std::ofstream file_stream(filename.c_str());
      tensor->tensor_->print(file_stream);
      file_stream.close();
    }
  }
#endif

  return status;
}
base::Status SequentialRuntime::postRun() {
  base::Status status = base::kStatusCodeOk;
  for (auto iter : op_repository_) {
    status = iter->op_->postRun();
    if (status != base::kStatusCodeOk) {
      NNDEPLOY_LOGE("Node %s postRun failed\n", iter->op_->getName().c_str());
      return status;
    }
  }
  // NNDEPLOY_LOGI("postRun ok!\n");
  return status;
}

base::Status SequentialRuntime::copyToInputTensor(device::Tensor* tensor) {
  if (!tensor) return base::kStatusCodeErrorNullParam;
  device::Tensor* src_tensor = tensor;
  device::Tensor* dst_tensor = nullptr;
  for (auto input : input_tensors_) {
    // NNDEPLOY_LOGI("copyToInputTensor input[%s].\n",
    // input->getName().c_str()); NNDEPLOY_LOGI("copyToInputTensor
    // src_tensor[%s].\n",
    //               src_tensor->getName().c_str());
    if (input->getName() == src_tensor->getName()) {
      dst_tensor = input;
      break;
    }
  }
  if (dst_tensor == nullptr) {
    NNDEPLOY_LOGE("copyToInputTensor failed! input tensor not found!\n");
    return base::kStatusCodeErrorInvalidValue;
  }
  if (dst_tensor->getDataType() != src_tensor->getDataType())
    return base::kStatusCodeErrorInvalidParam;
  if (dst_tensor->getShape() != src_tensor->getShape()) {
    // A pool resize may replace every input buffer, including inputs already copied.
    std::vector<std::pair<device::Tensor*, device::Tensor>> other_inputs;
    for (auto *input : input_tensors_) {
      if (input != dst_tensor && input->getBuffer())
        other_inputs.emplace_back(input, *input);
    }
    base::ShapeMap changed{{dst_tensor->getName(), src_tensor->getShape()}};
    auto status = reshape(changed);
    if (status != base::kStatusCodeOk) return status;
    for (auto &saved : other_inputs) {
      if (saved.second.getData() != saved.first->getData()) {
        status = saved.second.copyTo(saved.first);
        if (status != base::kStatusCodeOk) return status;
      }
    }
  }
  if (!dst_tensor->getBuffer() && is_pure_dynamic_shape_)
    dst_tensor->allocate(device::getDevice(device_type_));
  if (!dst_tensor->getBuffer() || !src_tensor->getBuffer())
    return base::kStatusCodeErrorInvalidValue;
  const size_t bytes = dst_tensor->getDevice()->toBufferDesc(dst_tensor->getDesc(), {}).getSize();
  if (src_tensor->getSize() < bytes || dst_tensor->getSize() < bytes)
    return base::kStatusCodeErrorInvalidValue;
  if (bytes && src_tensor->getData() != dst_tensor->getData()) {
    base::Status status;
    if (src_tensor->getSize() == bytes || dst_tensor->getSize() == bytes) {
      status = src_tensor->copyTo(dst_tensor);
    } else {
      // Both buffers contain spare capacity. Limit the copy to the input payload.
      device::Buffer view(src_tensor->getDevice(), bytes, src_tensor->getData());
      device::Tensor payload(src_tensor->getDesc(), &view);
      status = payload.copyTo(dst_tensor);
    }
    NNDEPLOY_RETURN_ON_NEQ(
        status, base::kStatusCodeOk,
        "copy external_input_tensor to internal_input_tensor failed!");
  }
  return base::kStatusCodeOk;
}

device::Tensor* SequentialRuntime::getOutputTensorAfterRun(
    const std::string& name, base::DeviceType device_type, bool is_copy,
    base::DataFormat data_format) {
  device::Device* device = device::getDevice(device_type);
  device::Tensor* internal_output_tensor = nullptr;
  for (auto output : output_tensors_) {
    if (output->getName() == name) {
      internal_output_tensor = output;
      break;
    }
  }
  if (internal_output_tensor == nullptr) {
    NNDEPLOY_LOGE("Not exsit output[%s].\n", name.c_str());
    return nullptr;
  }
  bool flag = is_copy || (internal_output_tensor->getDevice() != device);
  device::Tensor* output_tensor = nullptr;
  if (flag) {
    output_tensor =
        new device::Tensor(device, internal_output_tensor->getDesc(), name);
    // internal_tensor->getDesc().print();
    // output_tensor->getDesc().print();
    internal_output_tensor->copyTo(output_tensor);
    return output_tensor;
  } else {
    return internal_output_tensor;
  }
}

}  // namespace net
}  // namespace nndeploy