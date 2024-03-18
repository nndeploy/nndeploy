
#include "nndeploy/inference/onnxruntime/onnxruntime_inference.h"

#include "nndeploy/base/shape.h"
#include "nndeploy/inference/onnxruntime/onnxruntime_convert.h"
#include "nndeploy/inference/onnxruntime/onnxruntime_inference_param.h"

namespace nndeploy {
namespace inference {

TypeInferenceRegister<TypeInferenceCreator<OnnxRuntimeInference>>
    g_onnxruntime_inference_register(base::kInferenceTypeOnnxRuntime);

OnnxRuntimeInference::OnnxRuntimeInference(base::InferenceType type)
    : Inference(type) {}

OnnxRuntimeInference::~OnnxRuntimeInference() {
  session_ = Ort::Session{nullptr};
}

base::Status OnnxRuntimeInference::init() {
  base::Status status = base::kStatusCodeOk;

  is_share_command_queue_ = true;

  std::string model_buffer;
  OnnxRuntimeInferenceParam *onnxruntime_inference_param =
      dynamic_cast<OnnxRuntimeInferenceParam *>(inference_param_);
  if (onnxruntime_inference_param->is_path_) {
    model_buffer = base::openFile(onnxruntime_inference_param->model_value_[0]);
  } else {
    model_buffer = onnxruntime_inference_param->model_value_[0];
  }

  OnnxRuntimeConvert::convertFromInferenceParam(*onnxruntime_inference_param,
                                                session_options_);
  session_ = {env_, model_buffer.data(), model_buffer.size(), session_options_};

  binding_ = std::make_shared<Ort::IoBinding>(session_);
  auto memory_info =
      Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
  Ort::Allocator allocator(session_, memory_info);
  device::Device *device = device::getDefaultHostDevice();

  auto n_inputs = session_.GetInputCount();
  for (int i = 0; i < n_inputs; ++i) {
    auto input_name = session_.GetInputNameAllocated(i, allocator).get();

    auto type_info = session_.GetInputTypeInfo(i);
    std::vector<int64_t> shape =
        type_info.GetTensorTypeAndShapeInfo().GetShape();
    ONNXTensorElementDataType data_type =
        type_info.GetTensorTypeAndShapeInfo().GetElementType();
    inputs_desc_.emplace_back(OrtValueInfo{input_name, shape, data_type});

    device::TensorDesc desc;
    if (onnxruntime_inference_param->max_shape_.find(input_name) !=
        onnxruntime_inference_param->max_shape_.end()) {
      desc.shape_ = OnnxRuntimeConvert::convertToShape(
          shape, onnxruntime_inference_param->max_shape_[input_name]);
    } else {
      desc.shape_ = OnnxRuntimeConvert::convertToShape(shape);
    }
    desc.data_type_ = OnnxRuntimeConvert::convertToDataType(data_type);
    desc.data_format_ = OnnxRuntimeConvert::getDataFormatByShape(desc.shape_);
    device::Tensor *max_input_tensor =
        new device::Tensor(device, desc, input_name);
    max_input_tensors_.insert({input_name, max_input_tensor});

    device::Buffer *max_input_buffer = max_input_tensor->getBuffer();
    device::Tensor *current_input_tensor =
        new device::Tensor(desc, max_input_buffer, input_name);
    input_tensors_.insert({input_name, current_input_tensor});
  }

  for (auto iter : input_tensors_) {
    batch_size_ = iter.second->getBatch();
  }

  auto n_outputs = session_.GetOutputCount();
  for (int i = 0; i < n_outputs; ++i) {
    auto output_name = session_.GetOutputNameAllocated(i, allocator).get();

    auto type_info = session_.GetOutputTypeInfo(i);
    std::vector<int64_t> shape =
        type_info.GetTensorTypeAndShapeInfo().GetShape();
    ONNXTensorElementDataType data_type =
        type_info.GetTensorTypeAndShapeInfo().GetElementType();
    outputs_desc_.emplace_back(OrtValueInfo{output_name, shape, data_type});

    if (!isDynamic(shape)) {
      device::TensorDesc desc;
      desc.shape_ = OnnxRuntimeConvert::convertToShape(shape);
      desc.shape_[0] = batch_size_;
      desc.data_type_ = OnnxRuntimeConvert::convertToDataType(data_type);
      desc.data_format_ = OnnxRuntimeConvert::getDataFormatByShape(desc.shape_);
      device::Tensor *max_output_tensor =
          new device::Tensor(device, desc, output_name);
      max_output_tensors_.insert({output_name, max_output_tensor});

      device::Buffer *max_output_buffer = max_output_tensor->getBuffer();
      device::Tensor *current_output_tensor =
          new device::Tensor(desc, max_output_buffer, output_name);
      output_tensors_.insert({output_name, current_output_tensor});
    } else {
      device::Tensor *max_output_tensor = new device::Tensor(output_name);
      max_output_tensors_.insert({output_name, max_output_tensor});

      device::Tensor *current_output_tensor = new device::Tensor(output_name);
      output_tensors_.insert({output_name, current_output_tensor});
    }
  }

  return status;
}

base::Status OnnxRuntimeInference::deinit() {
  base::Status status = base::kStatusCodeOk;
  for (int i = 0; i < internal_outputs_.size(); ++i) {
    internal_outputs_[i].release();
  }
  internal_outputs_.clear();
  for (auto iter : input_tensors_) {
    delete iter.second;
  }
  input_tensors_.clear();
  for (auto iter : max_input_tensors_) {
    delete iter.second;
  }
  max_input_tensors_.clear();
  for (auto iter : output_tensors_) {
    delete iter.second;
  }
  output_tensors_.clear();
  for (auto iter : max_output_tensors_) {
    delete iter.second;
  }
  max_output_tensors_.clear();
  return status;
}

base::Status OnnxRuntimeInference::reshape(base::ShapeMap &shape_map) {
  auto memory_info =
      Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
  Ort::Allocator allocator(session_, memory_info);
  device::Device *device = device::getDefaultHostDevice();

  base::ShapeMap current_shape;
  auto n_inputs = session_.GetInputCount();
  for (int i = 0; i < n_inputs; ++i) {
    auto input_name = session_.GetInputNameAllocated(i, allocator).get();
    auto type_info = session_.GetInputTypeInfo(i);
    auto src_shape = type_info.GetTensorTypeAndShapeInfo().GetShape();
    auto shape = OnnxRuntimeConvert::convertToShape(src_shape);
    current_shape.insert({input_name, shape});
  }
  for (auto iter : shape_map) {
    auto tmp = current_shape.find(iter.first);
    if (tmp != current_shape.end()) {
      auto &shape = current_shape[iter.first];
      if (base::shapeEqual(iter.second, shape)) {
        continue;
      } else {
        device::TensorDesc desc = input_tensors_[iter.first]->getDesc();
        desc.shape_ = iter.second;
        input_tensors_[iter.first]->justModify(desc);
      }
    } else {
      NNDEPLOY_LOGE("reshape failed, not found input tensor(%s)!\n",
                    iter.first.c_str());
      return base::kStatusCodeErrorInferenceOnnxRuntime;
    }
  }

  return base::kStatusCodeOk;
}

base::Status OnnxRuntimeInference::run() {
  base::Status status = base::kStatusCodeOk;
  auto memory_info =
      Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
  Ort::Allocator allocator(session_, memory_info);
  device::Device *device = device::getDefaultHostDevice();
  try {
    auto n_inputs = session_.GetInputCount();
    for (int i = 0; i < n_inputs; ++i) {
      auto input_name = session_.GetInputNameAllocated(i, allocator).get();
      device::Tensor *input_tensor = nullptr;
      if (external_input_tensors_.find(input_name) !=
          external_input_tensors_.end()) {
        input_tensor = external_input_tensors_[input_name];
      } else {
        input_tensor = input_tensors_[input_name];
      }
      auto ort_value = OnnxRuntimeConvert::convertFromTensor(input_tensor);
      binding_->BindInput(input_name, ort_value);
    }

    for (size_t i = 0; i < outputs_desc_.size(); ++i) {
      Ort::MemoryInfo memory_info("Cpu", OrtDeviceAllocator, 0,
                                  OrtMemTypeDefault);
      binding_->BindOutput(outputs_desc_[i].name.c_str(), memory_info);
    }

    session_.Run({}, *binding_);

    for (int i = 0; i < internal_outputs_.size(); ++i) {
      internal_outputs_[i].release();
    }
    internal_outputs_.clear();
    // internal_outputs_ = std::move(binding_->GetOutputValues());
    internal_outputs_ = binding_->GetOutputValues();
  } catch (const std::exception &e) {
    NNDEPLOY_LOGE("%s.\n", e.what());
    status = base::kStatusCodeErrorInferenceOnnxRuntime;
  }
  return status;
}

device::Tensor *OnnxRuntimeInference::getOutputTensorAfterRun(
    const std::string &name, base::DeviceType device_type, bool is_copy,
    base::DataFormat data_format) {
  device::Tensor *external_output_tensor = nullptr;
  bool flag = is_copy || (!device::isHostDeviceType(device_type));
  device::Device *device = device::getDevice(device_type);
  for (size_t i = 0; i < internal_outputs_.size(); ++i) {
    auto output_name = outputs_desc_[i].name;
    if (output_name != name) {
      continue;
    }
    device::Tensor *output_tensor = output_tensors_[output_name];
    OnnxRuntimeConvert::convertToTensor(internal_outputs_[i], output_name,
                                        device, output_tensor);
    device::TensorDesc desc = output_tensor->getDesc();
    if (flag) {
      external_output_tensor = new device::Tensor(device, desc, name);
      device->copy(output_tensor->getBuffer(),
                   external_output_tensor->getBuffer());
      return external_output_tensor;
    } else {
      external_output_tensor =
          new device::Tensor(desc, output_tensor->getBuffer(), name);
      return external_output_tensor;
    }
  }
  return external_output_tensor;
}

bool OnnxRuntimeInference::isDynamic(std::vector<int64_t> &shape) {
  int size = shape.size();
  for (int i = 1; i < size; ++i) {
    if (shape[i] < 0) {
      return true;
    }
  }
  return false;
}

}  // namespace inference
}  // namespace nndeploy
