
#include "nndeploy/source/inference/mnn/mnn_inference.h"

namespace nndeploy {
namespace inference {

TypeInferenceRegister<TypeInferenceCreator<MnnInference>>
    g_internal_inference_register(base::kInferenceTypeMnn);

MnnInference::MnnInference(base::InferenceType type) : Inference(type) {
  internal_interpreter_ = nullptr;
  internal_session_ = nullptr;
}

MnnInference::~MnnInference() {}

base::Status MnnInference::init() {
  /**
   * @brief
   * @note
   * # InferenceParam -> MNN::ScheduleConfig
   * # 模型解析
   * # 能不能写入静态形状？
   */
  base::Status status = base::kStatusCodeOk;

  if (inference_param_->is_path_) {
    internal_interpreter_ = MNN::Interpreter::createFromFile(
        inference_param_->model_value_[0].c_str());
  } else {
    internal_interpreter_ = MNN::Interpreter::createFromBuffer(
        inference_param_->model_value_[0].c_str(),
        inference_param_->model_value_[0].length());
  }

  if (internal_interpreter_ == nullptr) {
    return base::kStatusCodeErrorInferenceMnn;
  }

  min_shape_ = inference_param_->min_shape_;
  opt_shape_ = inference_param_->opt_shape_;
  max_shape_ = inference_param_->max_shape_;

  MnnInferenceParam *mnn_inference_param =
      dynamic_cast<MnnInferenceParam *>(inference_param_);
  internal_inference_param_ = new MNN::ScheduleConfig();
  status = MnnConvert::convertFromInferenceParam(mnn_inference_param,
                                                 internal_inference_param_);
  NNDEPLOY_RETURN_ON_NEQ(status, base::kStatusCodeOk);
  internal_current_output_type = MnnConvert::convertFromDataFormat(
      mnn_inference_param->output_data_format_);

  internal_session_ =
      internal_interpreter_->createSession(*internal_inference_param_);

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

    device::Device *device = device::getDevice(inference_param_->device_type_);
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

    device::Device *device = device::getDevice(inference_param_->device_type_);
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

base::Status MnnInference::deinit() {
  if (internal_inference_param_ != nullptr) {
    delete internal_inference_param_;
  }

  bool third_status = internal_interpreter_->releaseSession(internal_session_);
  if (third_status) {
    return base::kStatusCodeOk;
  } else {
    return base::kStatusCodeErrorInferenceMnn;
  }

  if (internal_interpreter_ != nullptr) {
    MNN::Interpreter::destroy(internal_interpreter_);
  }
  return base::kStatusCodeOk;
}

base::Status MnnInference::reShape(base::ShapeMap &shape_map) {
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

int64_t MnnInference::getMemorySize() {
  MNN::Interpreter::SessionInfoCode code =
      MNN::Interpreter::SessionInfoCode::MEMORY;
  float fsize = 0;
  bool third_status =
      internal_interpreter_->getSessionInfo(internal_session_, code, &fsize);
  int64_t size = (int64_t)(fsize * 1024 * 1024);
  return size;
}

float MnnInference::getGFLOPs() { return 0.0f; }

base::Status MnnInference::setInputTensor(
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
std::shared_ptr<device::Tensor> MnnInference::getOutputTensor(
    const std::string &name, std::vector<int32_t> mnn_inference_param) {
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

  device::Device *device = device::getDevice(inference_param_->device_type_);
  tensors.reset(
      MnnConvert::convertToTensor(external_output_tensor, char_name, device));
  return tensors;
}

base::Status MnnInference::run() {
  MNN::ErrorCode third_status =
      internal_interpreter_->runSession(internal_session_);
  if (third_status != MNN::NO_ERROR) {
    return base::kStatusCodeErrorInferenceMnn;
  } else {
    return base::kStatusCodeOk;
  }
}

MNN::ScheduleConfig *MnnInference::getInternalInferenceParam() {
  return internal_inference_param_;
}

MNN::Interpreter *MnnInference::getInternalInterpreter() {
  return internal_interpreter_;
}

MNN::Session *MnnInference::getInternalSession() { return internal_session_; }

}  // namespace inference
}  // namespace nndeploy
