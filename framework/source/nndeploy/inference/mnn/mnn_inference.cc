
#include "nndeploy/inference/mnn/mnn_inference.h"

namespace nndeploy {
namespace inference {

TypeInferenceRegister<TypeInferenceCreator<MnnInference>>
    g_mnn_inference_register(base::kInferenceTypeMnn);

MnnInference::MnnInference(base::InferenceType type) : Inference(type) {
  schedule_config_ = nullptr;
  interpreter_ = nullptr;
  session_ = nullptr;
}
MnnInference::~MnnInference() {}

base::Status MnnInference::init() {
  base::Status status = base::kStatusCodeOk;

  if (device::isHostDeviceType(inference_param_->device_type_)) {
    is_external_stream_ = true;
  } else {
    is_external_stream_ = false;
  }

  if (inference_param_->is_path_) {
    interpreter_ = MNN::Interpreter::createFromFile(
        inference_param_->model_value_[0].c_str());
  } else {
    interpreter_ = MNN::Interpreter::createFromBuffer(
        inference_param_->model_value_[0].c_str(),
        inference_param_->model_value_[0].length());
  }
  if (interpreter_ == nullptr) {
    return base::kStatusCodeErrorInferenceMnn;
  }

  MnnInferenceParam *mnn_inference_param =
      dynamic_cast<MnnInferenceParam *>(inference_param_);
  schedule_config_ = new MNN::ScheduleConfig();
  status = MnnConvert::convertFromInferenceParam(mnn_inference_param,
                                                 schedule_config_);
  NNDEPLOY_RETURN_ON_NEQ(status, base::kStatusCodeOk, "init failed!\n");

  session_ = interpreter_->createSession(*schedule_config_);
  if (session_ == nullptr) {
    NNDEPLOY_LOGE("%s\n", "createSession failed");
    return base::kStatusCodeErrorInferenceMnn;
  }

  bool reshape_flag = false;
  const std::map<std::string, MNN::Tensor *> &tmp_internal_input_tensors =
      interpreter_->getSessionInputAll(session_);
  for (auto iter : tmp_internal_input_tensors) {
    std::string name = iter.first;
    base::IntVector dims = iter.second->shape();
    auto max_shape = inference_param_->max_shape_.find(name);
    if (max_shape != inference_param_->max_shape_.end()) {
      if (base::shapeEqual(max_shape->second, dims)) {
        continue;
      } else {
        // shape_的修改
        MNN::Tensor *tensor =
            interpreter_->getSessionInput(session_, iter.first.c_str());
        interpreter_->resizeTensor(tensor, max_shape->second);
        reshape_flag = true;
      }
    }
  }
  if (reshape_flag) {
    interpreter_->resizeSession(session_);
  }

  status = allocateInputOutputTensor();
  NNDEPLOY_RETURN_ON_NEQ(status, base::kStatusCodeOk,
                         "allocateInputOutputTensor failed!!\n");

  return status;
}

base::Status MnnInference::deinit() {
  base::Status status = deallocateInputOutputTensor();
  NNDEPLOY_RETURN_ON_NEQ(status, base::kStatusCodeOk,
                         "deallocateInputOutputTensor failed!!\n");

  if (schedule_config_ != nullptr) {
    delete schedule_config_;
  }

  bool third_status = interpreter_->releaseSession(session_);
  if (!third_status) {
    NNDEPLOY_LOGE("%s\n", "releaseSession failed");
    status = base::kStatusCodeErrorInferenceMnn;
  }

  if (interpreter_ != nullptr) {
    MNN::Interpreter::destroy(interpreter_);
  }

  return status;
}

base::Status MnnInference::reshape(base::ShapeMap &shape_map) {
  bool flag = false;
  for (auto iter : shape_map) {
    auto tmp = getInputShape(iter.first);
    if (!tmp.empty()) {
      if (base::shapeEqual(iter.second, tmp)) {
        continue;
      } else {
        MNN::Tensor *tensor =
            interpreter_->getSessionInput(session_, iter.first.c_str());
        interpreter_->resizeTensor(tensor, iter.second);
        flag = true;
      }
    } else {
      NNDEPLOY_LOGE("Not exsit input[%s].\n", iter.first.c_str());
      return base::kStatusCodeErrorInferenceMnn;
    }
  }

  if (flag) {
    interpreter_->resizeSession(session_);
    deallocateInputOutputTensor();
    allocateInputOutputTensor();
  }

  return base::kStatusCodeOk;
}

int64_t MnnInference::getMemorySize() {
  MNN::Interpreter::SessionInfoCode code =
      MNN::Interpreter::SessionInfoCode::MEMORY;
  float fsize = 0;
  bool third_status = interpreter_->getSessionInfo(session_, code, &fsize);
  int64_t size = (int64_t)(fsize * 1024 * 1024);
  return size;
}

float MnnInference::getGFLOPs() {
  MNN::Interpreter::SessionInfoCode code =
      MNN::Interpreter::SessionInfoCode::FLOPS;
  float flops = 0;
  bool third_status = interpreter_->getSessionInfo(session_, code, &flops);
  return flops / 1000.f;
}

device::TensorDesc MnnInference::getInputTensorAlignDesc(
    const std::string &name) {
  if (input_tensors_.count(name) > 0) {
    device::TensorDesc desc = input_tensors_[name]->getDesc();
    if (desc.shape_.size() == 5) {
      if (desc.data_format_ != base::kDataFormatNCDHW &&
          desc.data_format_ != base::kDataFormatNDHWC) {
        desc.data_format_ = base::kDataFormatNCDHW;
      }
    } else if (desc.shape_.size() == 4) {
      if (desc.data_format_ != base::kDataFormatNHWC &&
          desc.data_format_ != base::kDataFormatNCHW) {
        desc.data_format_ = base::kDataFormatNCHW;
      }
    } else if (desc.shape_.size() == 3) {
      if (desc.data_format_ != base::kDataFormatNCL) {
        desc.data_format_ = base::kDataFormatNCL;
      }
    } else if (desc.shape_.size() == 2) {
      if (desc.data_format_ != base::kDataFormatNC) {
        desc.data_format_ = base::kDataFormatNC;
      }
    } else if (desc.shape_.size() == 1) {
      if (desc.data_format_ != base::kDataFormatN) {
        desc.data_format_ = base::kDataFormatN;
      }
    } else {
      desc.data_format_ = base::kDataFormatNotSupport;
    }
    return desc;
  } else {
    return device::TensorDesc();
  }
}
device::TensorDesc MnnInference::getOutputTensorAlignDesc(
    const std::string &name) {
  if (output_tensors_.count(name) > 0) {
    device::TensorDesc desc = output_tensors_[name]->getDesc();
    if (desc.shape_.size() == 5) {
      if (desc.data_format_ != base::kDataFormatNCDHW &&
          desc.data_format_ != base::kDataFormatNDHWC) {
        desc.data_format_ = base::kDataFormatNCDHW;
      }
    } else if (desc.shape_.size() == 4) {
      if (desc.data_format_ != base::kDataFormatNHWC &&
          desc.data_format_ != base::kDataFormatNCHW) {
        desc.data_format_ = base::kDataFormatNCHW;
      }
    } else if (desc.shape_.size() == 3) {
      if (desc.data_format_ != base::kDataFormatNCL) {
        desc.data_format_ = base::kDataFormatNCL;
      }
    } else if (desc.shape_.size() == 2) {
      if (desc.data_format_ != base::kDataFormatNC) {
        desc.data_format_ = base::kDataFormatNC;
      }
    } else if (desc.shape_.size() == 1) {
      if (desc.data_format_ != base::kDataFormatN) {
        desc.data_format_ = base::kDataFormatN;
      }
    } else {
      desc.data_format_ = base::kDataFormatNotSupport;
    }
    return desc;
  } else {
    return device::TensorDesc();
  }
}

base::Status MnnInference::run() {
  // inputs
  for (auto iter : external_input_tensors_) {
    MNN::Tensor *external_tensor = MnnConvert::convertFromTensor((iter.second));
    if (external_tensor == nullptr) {
      NNDEPLOY_LOGE("convertFromTensor failed.\n");
      return base::kStatusCodeErrorInferenceMnn;
    }
    MNN::Tensor *internal_tensor =
        interpreter_->getSessionInput(session_, iter.first.c_str());
    if (internal_tensor == nullptr) {
      NNDEPLOY_LOGE("interpreter_->getSessionInput failed.\n");
      return base::kStatusCodeErrorInferenceMnn;
    }
    internal_tensor->copyFromHostTensor(external_tensor);
    delete external_tensor;
  }
  // forward
  MNN::ErrorCode third_status = interpreter_->runSession(session_);
  if (third_status != MNN::NO_ERROR) {
    NNDEPLOY_LOGE("interpreter_->runSessio failed.\n");
    return base::kStatusCodeErrorInferenceMnn;
  }
  // outputs
  // for (auto iter : external_output_tensors_) {
  //   MNN::Tensor *internal_tensor =
  //       interpreter_->getSessionOutput(session_, iter.first.c_str());
  //   if (internal_tensor == nullptr) {
  //     NNDEPLOY_LOGE("iinterpreter_->getSessionOutput failed.\n");
  //     return base::kStatusCodeErrorInferenceMnn;
  //   }
  //   MNN::Tensor *external_tensor =
  //   MnnConvert::convertFromTensor(iter.second);
  //   internal_tensor->copyToHostTensor(external_tensor);
  //   delete external_tensor;
  // }
  return base::kStatusCodeOk;
}

device::Tensor *MnnInference::getOutputTensorAfterRun(
    const std::string &name, base::DeviceType device_type, bool is_copy,
    base::DataFormat data_format) {
  MNN::Tensor *internal_tensor =
      interpreter_->getSessionOutput(session_, name.c_str());
  if (internal_tensor == nullptr) {
    NNDEPLOY_LOGE("iinterpreter_->getSessionOutput failed.\n");
    return nullptr;
  }
  bool can_op_flag = internal_tensor->getDimensionType() !=
                     MNN::Tensor::DimensionType::CAFFE_C4;
  can_op_flag = can_op_flag && is_external_stream_;
  device::Device *device = device::getDefaultHostDevice();
  if (is_copy || !can_op_flag) {
    device::TensorDesc desc = this->getInputTensorAlignDesc(name);
    device::Tensor *output_tensor = new device::Tensor(device, desc, name);
    MNN::Tensor *external_tensor = MnnConvert::convertFromTensor(output_tensor);
    internal_tensor->copyToHostTensor(external_tensor);
    delete external_tensor;
    return output_tensor;
  } else {
    device::Tensor *output_tensor =
        MnnConvert::convertToTensor(internal_tensor, name, device);
    return output_tensor;
  }
}

base::Status MnnInference::allocateInputOutputTensor() {
  device::Device *device = nullptr;
  if (device::isHostDeviceType(inference_param_->device_type_)) {
    device = device::getDevice(inference_param_->device_type_);
  }

  const std::map<std::string, MNN::Tensor *> &internal_input_tensors =
      interpreter_->getSessionInputAll(session_);
  for (auto iter : internal_input_tensors) {
    std::string name = iter.first;
    MNN::Tensor *internal_input_tensor = iter.second;

    device::Tensor *input_tensor =
        MnnConvert::convertToTensor(internal_input_tensor, name, device);
    input_tensors_.insert({name, input_tensor});
  }

  const std::map<std::string, MNN::Tensor *> &internal_output_tensors =
      interpreter_->getSessionOutputAll(session_);
  for (auto iter : internal_output_tensors) {
    std::string name = iter.first;
    MNN::Tensor *internal_output_tensor = iter.second;

    device::Tensor *output_tensor =
        MnnConvert::convertToTensor(internal_output_tensor, name, device);
    output_tensors_.insert({name, output_tensor});
  }
  return base::kStatusCodeOk;
}

base::Status MnnInference::deallocateInputOutputTensor() {
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
