

#include "nndeploy/net/ascend_cl/ascend_cl_session.h"

namespace nndeploy {
namespace net {

TypeSessionRegister<TypeSessionCreator<AscendCLSession>>
    g_ascend_cl_session_register_seq(
        base::DeviceTypeCode::kDeviceTypeCodeAscendCL,
        base::ParallelType::kParallelTypeSequential);

TypeSessionRegister<TypeSessionCreator<AscendCLSession>>
    g_ascend_cl_session_register_none(
        base::DeviceTypeCode::kDeviceTypeCodeAscendCL,
        base::ParallelType::kParallelTypeNone);

AscendCLSession::AscendCLSession(const base::DeviceType &device_type)
    : Session(device_type){};
AscendCLSession::~AscendCLSession(){};

base::Status AscendCLSession::init(
    std::vector<TensorWrapper *> &tensor_repository,
    std::vector<OpWrapper *> &op_repository, bool is_dynamic_shape,
    base::ShapeMap max_shape) {
  base::Status status = base::kStatusCodeOk;
  device::Device *device = device::getDevice(device_type_);
  // # 激活值的tensor分配
  tensor_pool_ = std::make_shared<TensorPool1DSharedObjectGreedyBySizeImprove>(
      device, tensor_repository, op_repository);
  /**
   * @brief
   * 如果是动态shape且max_shape为空时，那么不需要分配tensor
   */
  bool flag = is_dynamic_shape && max_shape.empty();
  if (!flag) {
    for (auto tensor_wrapper : tensor_repository) {
      auto shape_size = tensor_wrapper->tensor_->getShape().size();
      auto tensor = tensor_wrapper->tensor_;
      if (shape_size == 5) {
        tensor->setDataFormat(base::DataFormat::kDataFormatNCDHW);
      } else if (shape_size == 4) {
        if (tensor_wrapper->is_weight_) {
          tensor->setDataFormat(base::DataFormat::kDataFormatOIHW);
        } else {
          tensor->setDataFormat(base::DataFormat::kDataFormatNCHW);
        }
      } else if (shape_size == 3) {
        tensor->setDataFormat(base::DataFormat::kDataFormatNCW);
      } else if (shape_size == 2) {
        tensor->setDataFormat(base::DataFormat::kDataFormatNC);
      } else if (shape_size == 1) {
        tensor->setDataFormat(base::DataFormat::kDataFormatN);
      }
      auto desc = tensor->getDesc();
      NNDEPLOY_LOGE("tensor name = %s.\n", tensor->getName().c_str());
      desc.print();
    }
    NNDEPLOY_LOGE("hello world\n");
    status = tensor_pool_->allocate();
    NNDEPLOY_LOGE("hello world\n");
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
base::Status AscendCLSession::deinit() {
  base::Status status = base::kStatusCodeOk;
  for (auto iter : op_repository_) {
    status = iter->op_->deinit();
    if (status != base::kStatusCodeOk) {
      NNDEPLOY_LOGE("Node %s init failed\n", iter->op_->getName().c_str());
      return status;
    }
    iter->op_->setInitializedFlag(false);
  }
  status = tensor_pool_->deallocate();
  if (status != base::kStatusCodeOk) {
    NNDEPLOY_LOGE("tensor_pool_ allocate failed\n");
    return status;
  }
  return status;
}

base::Status AscendCLSession::reshape(base::ShapeMap &shape_map) {
  base::Status status = base::kStatusCodeOk;
  bool channge_flag = false;
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
        tensor->reshape(shape);
        channge_flag = true;
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
    }
  }
  if (channge_flag) {
    if (is_reallocate) {
      status = tensor_pool_->deallocate();
      if (status != base::kStatusCodeOk) {
        NNDEPLOY_LOGE("tensor_pool_ allocate failed\n");
        return status;
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

base::Status AscendCLSession::preRun() {
  base::Status status = base::kStatusCodeOk;
  for (auto iter : op_repository_) {
    status = iter->op_->preRun();
    if (status != base::kStatusCodeOk) {
      NNDEPLOY_LOGE("Node %s preRun failed\n", iter->op_->getName().c_str());
      return status;
    }
  }
  return status;
}
base::Status AscendCLSession::run() {
  base::Status status = base::kStatusCodeOk;
  // 存在优化空间
  uint64_t max_workspace_size = 0;
  for (auto iter : op_repository_) {
    uint64_t workspace_size = iter->op_->getWorkspaceSize();
    if (workspace_size > max_workspace_size) {
      max_workspace_size = workspace_size;
    }
  }
  NNDEPLOY_LOGE("max_workspace_size is %d.\n",
                static_cast<int32_t>(max_workspace_size));
  device::Device *device = device::getDevice(device_type_);
  void *workspace = nullptr;
  if (max_workspace_size > 0) {
    workspace = device->allocate(max_workspace_size);
  }
  for (auto iter : op_repository_) {
    iter->op_->setWorkspace(workspace);
  }
  for (auto iter : op_repository_) {
    status = iter->op_->run();
    if (status != base::kStatusCodeOk) {
      NNDEPLOY_LOGE("Node %s run failed\n", iter->op_->getName().c_str());
      return status;
    }
  }
  if (workspace != nullptr) {
    device->deallocate(workspace);
  }
  return status;
}
base::Status AscendCLSession::postRun() {
  base::Status status = base::kStatusCodeOk;
  for (auto iter : op_repository_) {
    status = iter->op_->postRun();
    if (status != base::kStatusCodeOk) {
      NNDEPLOY_LOGE("Node %s postRun failed\n", iter->op_->getName().c_str());
      return status;
    }
  }
  return status;
}

}  // namespace net
}  // namespace nndeploy