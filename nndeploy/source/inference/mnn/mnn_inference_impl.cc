
#include "nndeploy/source/inference/mnn/mnn_inference_impl.h"

namespace nndeploy {
namespace inference {

TypeInferenceRegister<TypeInferenceCreator<MnnInferenceImpl>>
    g_internal_inference_register(base::kInferenceTypeMnn);

MnnInferenceImpl::MnnInferenceImpl() {
  internal_config_ = nullptr;
  internal_interpreter_ = nullptr;
  internal_session_ = nullptr;
}

MnnInferenceImpl::~MnnInferenceImpl() {}

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
  status = MnnConvert::convertFromConfig(config, internal_config_);
  NNDEPLOY_RETURN_ON_NEQ(status, base::kStatusCodeOk);
  internal_current_output_type =
      MnnConvert::convertFromDataFormat(config->output_data_format_);

  internal_session_ = internal_interpreter_->createSession(*internal_config_);

  const std::map<std::string, MNN::Tensor *> &tmp_internal_input_tensors =
      internal_interpreter_->getSessionInputAll(internal_session_);
  for (auto iter : tmp_internal_input_tensors) {
    std::string name = iter.first;
    base::IntVector dims = iter.second->shape();
    current_shape_.insert({name, dims});
  }

  status = reShape(max_shape_);
  NNDEPLOY_RETURN_ON_NEQ(status, base::kStatusCodeOk);

  const std::map<std::string, MNN::Tensor *> &internal_input_tensors =
      internal_interpreter_->getSessionInputAll(internal_session_);
  for (auto iter : internal_input_tensors) {
    std::string name = iter.first;
    MNN::Tensor *internal_input_tensor = iter.second;
    std::shared_ptr<device::Tensor> max_input_tensor;

    device::Device *device = getDevice();
    max_input_tensor.reset(
        MnnConvert::convertToTensor(internal_input_tensor, name, device));
    max_input_tensors_.insert({name, max_input_tensor});

    device::TensorImplDesc desc = max_input_tensor->getDesc();
    device::Buffer *max_input_buffer = max_input_tensor->getBuffer();
    std::shared_ptr<device::Tensor> current_input_tensor;
    current_input_tensor.reset(
        new device::Tensor(desc, max_input_buffer, name));
    current_input_tensors_.insert({name, current_input_tensor});
  }
  const std::map<std::string, MNN::Tensor *> &internal_output_tensors =
      internal_interpreter_->getSessionOutputAll(internal_session_);
  for (auto iter : internal_output_tensors) {
    std::string name = iter.first;
    MNN::Tensor *internal_output_tensor = iter.second;
    std::shared_ptr<device::Tensor> max_output_tensor;

    device::Device *device = getDevice();
    max_output_tensor.reset(
        MnnConvert::convertToTensor(internal_output_tensor, name, device));
    max_output_tensors_.insert({name, max_output_tensor});

    device::TensorImplDesc desc = max_output_tensor->getDesc();
    device::Buffer *max_input_buffer = max_output_tensor->getBuffer();
    std::shared_ptr<device::Tensor> current_output_tensor;
    current_output_tensor.reset(
        new device::Tensor(desc, max_input_buffer, name));
    current_output_tensors_.insert({name, current_output_tensor});

    MNN::Tensor *internal_current_output_tensor =
        new MNN::Tensor(internal_output_tensor, internal_current_output_type);
    internal_current_output_tensors_.insert(
        {name, internal_current_output_tensor});
  }

  return status;
}

base::Status MnnInferenceImpl::postRun() {
  if (internal_config_ != nullptr) {
    if (internal_config_->backendConfig != nullptr) {
      delete internal_config_->backendConfig;
    }
    delete internal_config_;
  }
  bool third_status = internal_interpreter_->releaseSession(internal_session_);
  if (third_status) {
    return base::kStatusCodeOk;
  } else {
    return base::kStatusCodeErrorInferenceMnn;
  }
}

base::Status MnnInferenceImpl::reShape(base::ShapeMap &shape_map) {
  bool flag = false;
  for (auto iter : shape_map) {
    auto tmp = current_shape_.find(iter.first);
    if (tmp != current_shape_.end()) {
      auto &shape = current_shape_[iter.first];
      if (base::shapeEqual(iter.second, shape)) {
        continue;
      } else {
        // current_shape_的修改
        current_shape_[iter.first] = iter.second;
        device::TensorImplDesc desc =
            current_input_tensors_[iter.first]->getDesc();
        desc.shape_ = iter.second;
        current_input_tensors_[iter.first]->justModify(desc);
        MNN::Tensor *tensor = internal_interpreter_->getSessionInput(
            internal_session_, iter.first.c_str());
        internal_interpreter_->resizeTensor(tensor, shape);
        flag = true;
      }
    } else {
      return base::kStatusCodeErrorInferenceMnn;
    }
  }

  if (flag) {
    internal_interpreter_->resizeSession(internal_session_);
    const std::map<std::string, MNN::Tensor *> &internal_output_tensors =
        internal_interpreter_->getSessionOutputAll(internal_session_);
    for (auto iter : internal_output_tensors) {
      std::string name = iter.first;
      MNN::Tensor *internal_output_tensor = iter.second;
      base::IntVector shape = internal_output_tensor->shape();
      device::TensorImplDesc desc = current_output_tensors_[name]->getDesc();
      desc.shape_ = shape;
      current_input_tensors_[iter.first]->justModify(desc);

      MNN::Tensor *internal_current_output_tensor =
          new MNN::Tensor(internal_output_tensor, internal_current_output_type);
      if (internal_current_output_tensors_.find(name) !=
          internal_current_output_tensors_.end()) {
        delete internal_current_output_tensors_[name];
        internal_current_output_tensors_[name] = internal_current_output_tensor;
      } else {
        internal_current_output_tensors_.insert(
            {name, internal_current_output_tensor});
      }
    }
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

float MnnInferenceImpl::getGFLOPs() { return 0.0f; }

base::Status MnnInferenceImpl::setInputTensor(
    const std::string &name,
    const std::shared_ptr<device::Tensor> input_tensor) {
  base::Status status = base::kStatusCodeOk;
  MNN::Tensor *external_input_tensor =
      MnnConvert::convertFromTensor(input_tensor.get());
  if (external_input_tensor == nullptr) {
    return base::kStatusCodeErrorInferenceMnn;
  }

  char *char_name = nullptr;
  if (!name.empty()) {
    char_name = const_cast<char *>(name.c_str());
  } else if (!input_tensor->getName().empty()) {
    char_name = const_cast<char *>(input_tensor->getName().c_str());
  }
  MNN::Tensor *internal_input_tensor =
      internal_interpreter_->getSessionInput(internal_session_, char_name);
  if (internal_input_tensor == nullptr) {
    return base::kStatusCodeErrorInferenceMnn;
  }
  internal_input_tensor->copyFromHostTensor(external_input_tensor);

  return status;
}
//
std::shared_ptr<device::Tensor> MnnInferenceImpl::getOutputTensor(
    const std::string &name, std::vector<int32_t> config) {
  std::shared_ptr<device::Tensor> tensors;
  char *char_name = nullptr;
  if (!name.empty()) {
    char_name = const_cast<char *>(name.c_str());
  }
  MNN::Tensor *internal_output_tensor =
      internal_interpreter_->getSessionOutput(internal_session_, char_name);
  if (internal_output_tensor == nullptr) {
    return tensors;
  }
  MNN::Tensor *external_output_tensor = internal_current_output_tensors_[name];
  internal_output_tensor->copyToHostTensor(external_output_tensor);

  device::Device *device = getDevice();
  tensors.reset(
      MnnConvert::convertToTensor(external_output_tensor, char_name, device));
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

}  // namespace inference
}  // namespace nndeploy
