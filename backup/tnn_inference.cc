
#include "nndeploy/inference/tnn/tnn_inference.h"

namespace nndeploy {
namespace inference {

TypeInferenceRegister<TypeInferenceCreator<TnnInference>>
    g_internal_inference_register(base::kInferenceTypeTnn);

TnnInference::TnnInference(base::InferenceType type) : Inference(type) {
  internal_inference_param_ = nullptr;
  internal_interpreter_ = nullptr;
  internal_session_ = nullptr;
}

TnnInference::~TnnInference() {}

base::Status TnnInference::init() {
  base::Status status = base::kStatusCodeOk;

  /**
   * @brief
   *
   */
  if (device::isHostDeviceType(inference_param_->device_type_)) {
    is_share_command_queue_ = true;
  } else {
    is_share_command_queue_ = false;
  }

  /**
   * @brief
   *
   */
  if (inference_param_->is_path_) {
    internal_interpreter_ = TNN::Interpreter::createFromFile(
        inference_param_->model_value_[0].c_str());
  } else {
    internal_interpreter_ = TNN::Interpreter::createFromBuffer(
        inference_param_->model_value_[0].c_str(),
        inference_param_->model_value_[0].length());
  }
  if (internal_interpreter_ == nullptr) {
    return base::kStatusCodeErrorInferenceTnn;
  }

  /**
   * @brief
   *
   */
  TnnInferenceParam *tnn_inference_param =
      dynamic_cast<TnnInferenceParam *>(inference_param_);
  internal_inference_param_ = new TNN::ScheduleConfig();
  status = TnnConvert::convertFromInferenceParam(tnn_inference_param,
                                                 internal_inference_param_);
  NNDEPLOY_RETURN_ON_NEQ(status, base::kStatusCodeOk, "init failed");

  /**
   * @brief
   *
   */
  internal_session_ =
      internal_interpreter_->createSession(*internal_inference_param_);

  /**
   * @brief
   *
   */
  bool reshape_flag = false;
  const std::map<std::string, TNN::Tensor *> &tmp_internal_input_tensors =
      internal_interpreter_->getSessionInputAll(internal_session_);
  for (auto iter : tmp_internal_input_tensors) {
    std::string name = iter.first;
    base::IntVector dims = iter.second->shape();
    auto max_shape = inference_param_->max_shape_.find(name);
    if (max_shape != inference_param_->max_shape_.end()) {
      if (base::shapeEqual(max_shape->second, dims)) {
        continue;
      } else {
        // shape_的修改
        TNN::Tensor *tensor = internal_interpreter_->getSessionInput(
            internal_session_, iter.first.c_str());
        internal_interpreter_->resizeTensor(tensor, max_shape->second);
        reshape_flag = true;
      }
    }
  }
  if (reshape_flag) {
    internal_interpreter_->resizeSession(internal_session_);
    const std::map<std::string, TNN::Tensor *> &internal_output_tensors =
        internal_interpreter_->getSessionOutputAll(internal_session_);
  }

  /**
   * @brief
   *
   */
  allocateInputOutputTensor();

  /**
   * @brief
   *
   */
  is_input_dynamic_ = inference_param_->is_dynamic_shape_;
  // TODO: 有可能输入非动态，但是输出是动态的
  is_output_dynamic_ = is_input_dynamic_;
  can_op_input_ = false;
  can_op_output_ = false;

  return status;
}

base::Status TnnInference::deinit() {
  deallocateInputOutputTensor();

  if (internal_inference_param_ != nullptr) {
    delete internal_inference_param_;
  }

  bool third_status = internal_interpreter_->releaseSession(internal_session_);
  if (!third_status) {
    NNDEPLOY_LOGE("%s\n", "releaseSession failed");
    return base::kStatusCodeErrorInferenceTnn;
  }

  if (internal_interpreter_ != nullptr) {
    TNN::Interpreter::destroy(internal_interpreter_);
  }
  return base::kStatusCodeOk;
}

base::Status TnnInference::reshape(base::ShapeMap &shape_map) {
  bool flag = false;
  for (auto iter : shape_map) {
    auto tmp = getInputShape(iter.first);
    if (!tmp.empty()) {
      if (base::shapeEqual(iter.second, tmp)) {
        continue;
      } else {
        TNN::Tensor *tensor = internal_interpreter_->getSessionInput(
            internal_session_, iter.first.c_str());
        internal_interpreter_->resizeTensor(tensor, tmp);
        flag = true;
      }
    } else {
      return base::kStatusCodeErrorInferenceTnn;
    }
  }

  if (flag) {
    internal_interpreter_->resizeSession(internal_session_);
    deallocateInputOutputTensor();
    allocateInputOutputTensor();
  }

  return base::kStatusCodeOk;
}

int64_t TnnInference::getMemorySize() {
  TNN::Interpreter::SessionInfoCode code =
      TNN::Interpreter::SessionInfoCode::MEMORY;
  float fsize = 0;
  bool third_status =
      internal_interpreter_->getSessionInfo(internal_session_, code, &fsize);
  int64_t size = (int64_t)(fsize * 1024 * 1024);
  return size;
}

float TnnInference::getGFLOPs() { return 0.0f; }

device::TensorDesc TnnInference::getInputTensorAlignDesc(
    const std::string &name) {
  TnnInferenceParam *tnn_inference_param =
      dynamic_cast<TnnInferenceParam *>(inference_param_);
  if (input_tensors_.count(name) > 0) {
    device::TensorDesc desc = input_tensors_[name]->getDesc();
    if (desc.shape_.size() == 4) {
      if (tnn_inference_param->inputs_data_format_ ==
          nndeploy::base::kDataFormatAuto) {
        desc.format_ = nndeploy::base::kDataFormatNCHW;
      }
    }
    return desc;
  } else {
    return device::TensorDesc();
  }
}
device::TensorDesc TnnInference::getOutputTensorAlignDesc(
    const std::string &name) {
  TnnInferenceParam *tnn_inference_param =
      dynamic_cast<TnnInferenceParam *>(inference_param_);
  if (output_tensors_.count(name) > 0) {
    device::TensorDesc desc = output_tensors_[name]->getDesc();
    if (desc.shape_.size() == 4) {
      if (tnn_inference_param->outputs_data_format_ ==
          nndeploy::base::kDataFormatNCHW) {
        desc.format_ = nndeploy::base::kDataFormatNCHW;
      }
    }
    return desc;
  } else {
    return device::TensorDesc();
  }
}

base::Status TnnInference::run() {
  // inputs
  for (auto iter : external_input_tensors_) {
    TNN::Tensor *external_tensor = TnnConvert::convertFromTensor((iter.second));
    if (external_tensor == nullptr) {
      NNDEPLOY_LOGE("convertFromTensor failed.\n");
      return base::kStatusCodeErrorInferenceTnn;
    }
    TNN::Tensor *internal_tensor = internal_interpreter_->getSessionInput(
        internal_session_, iter.first.c_str());
    if (internal_tensor == nullptr) {
      NNDEPLOY_LOGE("internal_interpreter_->getSessionInput failed.\n");
      return base::kStatusCodeErrorInferenceTnn;
    }
    internal_tensor->copyFromHostTensor(external_tensor);
    delete external_tensor;
  }
  // forward
  TNN::ErrorCode third_status =
      internal_interpreter_->runSession(internal_session_);
  if (third_status != TNN::NO_ERROR) {
    NNDEPLOY_LOGE("internal_interpreter_->runSessio failed.\n");
    return base::kStatusCodeErrorInferenceTnn;
  }
  // outputs
  for (auto iter : external_output_tensors_) {
    TNN::Tensor *internal_tensor = internal_interpreter_->getSessionOutput(
        internal_session_, iter.first.c_str());
    if (internal_tensor == nullptr) {
      NNDEPLOY_LOGE("iinternal_interpreter_->getSessionOutput failed.\n");
      return base::kStatusCodeErrorInferenceTnn;
    }
    TNN::Tensor *external_tensor = TnnConvert::convertFromTensor(iter.second);
    internal_tensor->copyToHostTensor(external_tensor);
    delete external_tensor;
  }
  return base::kStatusCodeOk;
}

TNN::ScheduleConfig *TnnInference::getInternalInferenceParam() {
  return internal_inference_param_;
}

TNN::Interpreter *TnnInference::getInternalInterpreter() {
  return internal_interpreter_;
}

TNN::Session *TnnInference::getInternalSession() { return internal_session_; }

base::Status TnnInference::allocateInputOutputTensor() {
  device::Device *device = nullptr;
  if (device::isHostDeviceType(inference_param_->device_type_)) {
    device = device::getDevice(inference_param_->device_type_);
  }
  const std::map<std::string, TNN::Tensor *> &internal_input_tensors =
      internal_interpreter_->getSessionInputAll(internal_session_);
  for (auto iter : internal_input_tensors) {
    std::string name = iter.first;
    TNN::Tensor *internal_input_tensor = iter.second;

    device::Tensor *input_tensor =
        TnnConvert::convertToTensor(internal_input_tensor, name, device);
    input_tensors_.insert({name, input_tensor});
  }
  const std::map<std::string, TNN::Tensor *> &internal_output_tensors =
      internal_interpreter_->getSessionOutputAll(internal_session_);
  for (auto iter : internal_output_tensors) {
    std::string name = iter.first;
    TNN::Tensor *internal_output_tensor = iter.second;

    device::Tensor *output_tensor =
        TnnConvert::convertToTensor(internal_output_tensor, name, device);
    output_tensors_.insert({name, output_tensor});
  }
  return base::kStatusCodeOk;
}

base::Status TnnInference::deallocateInputOutputTensor() {
  for (auto iter : input_tensors_) {
    delete iter.second;
  }
  input_tensors_.clear();
  for (auto iter : output_tensors_) {
    delete iter.second;
  }
  output_tensors_.clear();
  return base::kStatusCodeOk;
}

}  // namespace inference
}  // namespace nndeploy
