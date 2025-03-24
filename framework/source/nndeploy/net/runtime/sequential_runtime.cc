

#include "nndeploy/net/runtime/sequential_runtime.h"

#include "nndeploy/base/time_profiler.h"

namespace nndeploy {
namespace net {

TypeRuntimeRegister<TypeRuntimeCreator<SequentialRuntime>>
    g_sequential_runtime_register_seq(
        base::ParallelType::kParallelTypeSequential);

TypeRuntimeRegister<TypeRuntimeCreator<SequentialRuntime>>
    g_sequential_runtime_register_none(base::ParallelType::kParallelTypeNone);

SequentialRuntime::SequentialRuntime(const base::DeviceType &device_type)
    : Runtime(device_type) {};
SequentialRuntime::~SequentialRuntime() {};

void SequentialRuntime::setAllocateInputOutputTensor(
    bool allocate_input_output_tensor) {
  allocate_input_output_tensor_ = allocate_input_output_tensor;
}

base::Status SequentialRuntime::init(
    std::vector<TensorWrapper *> &tensor_repository,
    std::vector<OpWrapper *> &op_repository, bool is_dynamic_shape,
    base::ShapeMap max_shape, TensorPoolType tensor_pool_type) {
  base::Status status = base::kStatusCodeOk;
  device::Device *device = device::getDevice(device_type_);

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
  tensor_pool_->setAllocateInputOutputTensor(allocate_input_output_tensor_);
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
  base::Status status = base::kStatusCodeOk;
  for (auto iter : op_repository_) {
    status = iter->op_->deinit();
    if (status != base::kStatusCodeOk) {
      NNDEPLOY_LOGE("Node %s init failed\n", iter->op_->getName().c_str());
      return status;
    }
    iter->op_->setInitializedFlag(false);
  }
  bool flag = is_dynamic_shape_ && max_shape_.empty();
  if (!flag) {
    status = tensor_pool_->deallocate();
    if (status != base::kStatusCodeOk) {
      NNDEPLOY_LOGE("tensor_pool_ allocate failed\n");
      return status;
    }
  }
  delete tensor_pool_;
  return status;
}

// 可以性能优化
base::Status SequentialRuntime::reshape(base::ShapeMap &shape_map) {
  base::Status status = base::kStatusCodeOk;
  if (!is_dynamic_shape_) {
    NNDEPLOY_LOGE("reshape is not supported in static shape\n");
    return base::kStatusCodeErrorInvalidParam;
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
            for (int i = 0; i < shape.size(); ++i) {
              if (shape[i] > max_shape[i]) {
                is_reallocate = true;
                break;
              }
            }
          }
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
          tensor->reshape(shape);
        }
      }
    }
    for (auto iter : op_repository_) {
      status = iter->op_->inferShape();
      if (status != base::kStatusCodeOk) {
        NNDEPLOY_LOGE("Node %s init failed\n", iter->op_->getName().c_str());
        return status;
      }
    }
    if (is_reallocate) {
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
  device::Device *device = device::getDevice(device_type_);
  if (workspace_size_ == 0 || is_dynamic_shape_) {
    if (workspace_ != nullptr) {
      device->deallocate(workspace_);
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
    } else {
      workspace_size_ = -1;  // 所有算子不需要额外内存空间
    }
    for (auto iter : op_repository_) {
      iter->op_->setWorkspace(workspace_);
    }
  }

  NNDEPLOY_TIME_POINT_START("net->run()");
  for (auto iter : op_repository_) {
    status = iter->op_->run();
    NNDEPLOY_LOGE("Node %s run\n", iter->op_->getName().c_str());
    if (status != base::kStatusCodeOk) {
      NNDEPLOY_LOGE("Node %s run failed\n", iter->op_->getName().c_str());
      return status;
    }
  }
  NNDEPLOY_TIME_POINT_END("net->run()");

  // NNDEPLOY_LOGI("run ok!\n");

#if 1
  status = stream_->synchronize();
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

}  // namespace net
}  // namespace nndeploy