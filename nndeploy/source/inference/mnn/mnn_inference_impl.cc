
#include "nndeploy/source/inference/mnn/mnn_inference_impl.h"

#include "nndeploy/source/base/shape.h"

namespace nndeploy {
namespace inference {

TypeInferenceRegister<TypeInferenceCreator<MnnInferenceImpl>>
    g_internal_inference_register(base::kInferenceTypeMnn);

MnnInferenceImpl::MnnInferenceImpl() {
  internal_config_ = nullptr;
  internal_interpreter_ = nullptr;
  internal_session_ = nullptr;
}

MnnInferenceImpl::~MnnInferenceImpl() {
  if (internal_config_ != nullptr) {
    if (internal_config_->backendConfig != nullptr) {
      delete internal_config_->backendConfig;
    }
    delete internal_config_;
  }
}

base::Status MnnInferenceImpl::init(std::shared_ptr<Config> config) {
  /**
   * @brief
   * @note
   * # Config -> MNN::ScheduleConfig
   * # 模型解析
   * # 能不能写入静态形状？
   */
  base::Status status = base::kStatusCodeOk;

  config_ = config;
  if (config->config_impl_->is_path_) {
    internal_interpreter_ = MNN::Interpreter::createFromFile(
        config->config_impl_->model_value_[0].c_str());
  } else {
    internal_interpreter_ = MNN::Interpreter::createFromBuffer(
        config->config_impl_->model_value_[0].c_str(),
        config->config_impl_->model_value_[0].length());
  }

  if (internal_interpreter_ == nullptr) {
    return base::kStatusCodeErrorInferenceMnn;
  }

  return status;
}

base::Status MnnInferenceImpl::deinit() {
  if (internal_interpreter_ != nullptr) {
    MNN::Interpreter::destroy(internal_interpreter_);
  }
  return base::kStatusCodeOk;
}

base::Status MnnInferenceImpl::preRun(base::ShapeMap min_shape,
                                      base::ShapeMap opt_shape,
                                      base::ShapeMap max_shape) {
  base::Status status = base::kStatusCodeOk;

  min_shape_ = min_shape;
  opt_shape_ = opt_shape;
  max_shape_ = max_shape;

  MnnConfigImpl *config = dynamic_cast<MnnConfigImpl *>(config_->config_impl_);
  internal_config_ = new MNN::ScheduleConfig();
  status = convertInternalConfig(config, internal_config_);
  NNDEPLOY_RETURN_ON_NEQ(status, base::kStatusCodeOk);

  internal_session_ = internal_interpreter_->createSession(*internal_config_);
  current_shape_ = getInternalInputShapeMap();
  if (!max_shape.empty()) {
    reShape(max_shape);
  }

  status = createInternalInputsOutputs();
  NNDEPLOY_RETURN_ON_NEQ(status, base::kStatusCodeOk);

  return status;
}

base::Status MnnInferenceImpl::postRun() {
  bool third_status = internal_interpreter_->releaseSession(internal_session_);
  if (third_status) {
    return base::kStatusCodeOk;
  } else {
    return base::kStatusCodeErrorInferenceMnn;
  }
}

// TODO：需要更新update current shape 和 current input tensor
base::Status MnnInferenceImpl::reShape(base::ShapeMap &shape_map) {
  bool flag = false;
  for (auto iter : shape_map) {
    auto tmp = current_shape_.find(iter.first);
    if (tmp != current_shape_.end()) {
      auto &shape = current_shape_[iter.first];
      if (base::shapeEqual(iter.second, shape)) {
        continue;
      } else {
        // TODO：map的使用
        shape = iter.second;
        MNN::Tensor *tensor = internal_interpreter_->getSessionInput(
            internal_session_, iter.first.c_str());
        internal_interpreter_->resizeTensor(tensor, shape);
        flag = true;
        // TODO：current shape也需要修改
      }
    } else {
      // TODO：add log
      return base::kStatusCodeErrorInferenceMnn;
    }
  }

  if (flag) {
    internal_interpreter_->resizeSession(internal_session_);
  }

  return base::kStatusCodeOk;
}

int64_t MnnInferenceImpl::getMemorySize() {
  MNN::Interpreter::SessionInfoCode code =
      MNN::Interpreter::SessionInfoCode::MEMORY;
  float fsize = 0;
  bool third_status =
      internal_interpreter_->getSessionInfo(internal_session_, code, &fsize);
  int64_t size = (int64_t)(fsize * 1024 * 1024);
  return size;
}

base::Status MnnInferenceImpl::setInputTensor(
    const std::string &name,
    const std::shared_ptr<device::Tensor> input_tensor) {
  return base::kStatusCodeOk;
}
//
std::shared_ptr<device::Tensor> MnnInferenceImpl::getOutputTensor(
    const std::string &name, std::vector<int32_t> config) {
  std::shared_ptr<device::Tensor> tensors;

  return tensors;
}

base::Status MnnInferenceImpl::run() {
  MNN::ErrorCode third_status =
      internal_interpreter_->runSession(internal_session_);
  if (third_status != MNN::NO_ERROR) {
    return base::kStatusCodeErrorInferenceMnn;
  } else {
    return base::kStatusCodeOk;
  }
}
base::Status MnnInferenceImpl::asyncRun() { return this->run(); }

MNN::ScheduleConfig *MnnInferenceImpl::getInternalConfig() {
  return internal_config_;
}
MNN::Interpreter *MnnInferenceImpl::getInternalInterpreter() {
  return internal_interpreter_;
}
MNN::Session *MnnInferenceImpl::getInternalSession() {
  return internal_session_;
}

base::Status MnnInferenceImpl::convertInternalConfig() {
  if (internal_config_ == nullptr) {
    internal_config_ = new MNN::ScheduleConfig;
  }

  return base::kStatusCodeOk;
}

base::ShapeMap MnnInferenceImpl::getInternalInputShapeMap() {
  const std::map<std::string, MNN::Tensor *> &internal_input_tensors =
      internal_interpreter_->getSessionInputAll(internal_session_);
  base::ShapeMap shape;
  for (auto iter : internal_input_tensors) {
    std::string name = iter.first;
    base::IntVector dims = iter.second->shape();
    if (dims[0] != 1) {
      dims[0] = 1;
      internal_interpreter_->resizeTensor(iter.second, dims);
      internal_interpreter_->resizeSession(internal_session_);
    }
    shape.insert({name, dims});
  }
  return shape;
}

base::Status MnnInferenceImpl::createInternalInputsOutputs() {
  const std::map<std::string, MNN::Tensor *> &internal_input_tensors =
      internal_interpreter_->getSessionInputAll(internal_session_);
  for (auto iter : internal_input_tensors) {
    std::string name = iter.first;
    MNN::Tensor *internal_input_tensor = iter.second;
    std::shared_ptr<device::Tensor> max_input_tensor;
    max_input_tensor.reset(new device::Tensor());
    // TODO
    device::Device *device = DeviceManager::getInstance().getDevice("CPU");
    max_input_tensor.create();
    max_input_tensors_.insert({name, max_input_tensor});
    std::shared_ptr<device::Tensor> current_input_tensor;
    current_input_tensor.reset(new device::Tensor());
    // TODO
    // current_input_tensor.create();
    current_input_tensors_.insert({name, max_input_tensor});
  }
  const std::map<std::string, MNN::Tensor *> &internal_output_tensors =
      internal_interpreter_->getSessionOutputAll(internal_session_);
  for (auto iter : internal_output_tensors) {
    std::string name = iter.first;
    MNN::Tensor *internal_input_tensor = iter.second;
    std::shared_ptr<device::Tensor> max_output_tensor;
    max_output_tensor.reset(new device::Tensor());
    // TODO
    // max_output_tensor.create();
    max_output_tensors_.insert({name, max_output_tensor});
    std::shared_ptr<device::Tensor> current_output_tensor;
    current_output_tensor.reset(new device::Tensor());
    // TODO
    // current_output_tensor.create();
    current_output_tensors_.insert({name, max_output_tensor});
  }

  return base::kStatusCodeOk;
}

}  // namespace inference
}  // namespace nndeploy
