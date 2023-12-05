
#include "nndeploy/inference/openvino/openvino_inference.h"

#include "nndeploy/base/shape.h"
#include "nndeploy/inference/openvino/openvino_convert.h"
#include "nndeploy/inference/openvino/openvino_inference_param.h"

namespace nndeploy {
namespace inference {

TypeInferenceRegister<TypeInferenceCreator<OpenVinoInference>>
    g_openvino_inference_register(base::kInferenceTypeOpenVino);

ov::Core OpenVinoInference::core_;

OpenVinoInference::OpenVinoInference(base::InferenceType type)
    : Inference(type) {}

OpenVinoInference::~OpenVinoInference() {}

base::Status OpenVinoInference::init() {
  base::Status status = base::kStatusCodeOk;
  if (device::isHostDeviceType(inference_param_->device_type_)) {
    is_share_command_queue_ = true;
  } else {
    is_share_command_queue_ = false;
  }
  status = reshape(inference_param_->max_shape_);
  NNDEPLOY_RETURN_ON_NEQ(status, base::kStatusCodeOk, "init failed!\n");
  return status;
}

base::Status OpenVinoInference::deinit() {
  base::Status status = base::kStatusCodeOk;
  for (auto iter : input_tensors_) {
    delete iter.second;
  }
  input_tensors_.clear();
  for (auto iter : output_tensors_) {
    delete iter.second;
  }
  output_tensors_.clear();
  input_index_map_.clear();
  output_index_map_.clear();
  return status;
}

base::Status OpenVinoInference::reshape(base::ShapeMap &shape_map) {
  base::Status status = base::kStatusCodeOk;
  if (initialized_) {
    base::ShapeMap current_shape = getAllInputShape();
    bool is_same = true;
    for (auto iter : shape_map) {
      auto tmp = current_shape.find(iter.first);
      if (tmp != current_shape.end()) {
        auto &shape = current_shape[iter.first];
        if (base::shapeEqual(iter.second, shape)) {
          continue;
        } else {
          is_same = false;
          break;
        }
      } else {
        NNDEPLOY_LOGE("reshape failed, not found input tensor(%s)!\n",
                      iter.first.c_str());
        return base::kStatusCodeErrorInferenceOpenVino;
      }
    }
    if (is_same) {
      return base::kStatusCodeOk;
    }
  }

  input_index_map_.clear();
  output_index_map_.clear();

  OpenVinoInferenceParam *openvino_inference_param =
      dynamic_cast<OpenVinoInferenceParam *>(inference_param_);

  std::shared_ptr<ov::Model> model;
  if (openvino_inference_param->is_path_) {
    model = core_.read_model(openvino_inference_param->model_value_[0]);
  } else {
    if (openvino_inference_param->model_type_ == base::kModelTypeOnnx) {
      model = core_.read_model(openvino_inference_param->model_value_[0],
                               ov::Tensor());
    } else if (openvino_inference_param->model_type_ ==
               base::kModelTypeOpenVino) {
      ov::element::Type type(ov::element::u8);
      std::vector<size_t> axis_lengths = {
          openvino_inference_param->model_value_[1].size()};
      ov::Shape shape(axis_lengths);
      void *host_ptr = (void *)openvino_inference_param->model_value_[1].data();
      ov::Tensor tensor(type, shape, host_ptr);
      model =
          core_.read_model(openvino_inference_param->model_value_[0], tensor);
    } else {
      NNDEPLOY_LOGE("not support this model type(%d)!\n",
                    openvino_inference_param->model_type_);
      return base::kStatusCodeErrorInferenceOpenVino;
    }
  }
  if (openvino_inference_param->max_shape_.size() > 0) {
    std::map<std::string, ov::PartialShape> ov_shape;
    for (const auto &item : openvino_inference_param->max_shape_) {
      ov_shape[item.first] = OpenVinoConvert::convertFromShape(item.second);
    }
    model->reshape(ov_shape);
  }

  const std::vector<ov::Output<ov::Node>> inputs = model->inputs();
  for (int i = 0; i < inputs.size(); ++i) {
    std::string name = inputs[i].get_any_name();
    input_index_map_.insert(std::make_pair(name, i));
  }
  const std::vector<ov::Output<ov::Node>> outputs = model->outputs();
  for (int i = 0; i < outputs.size(); ++i) {
    std::string name = outputs[i].get_any_name();
    output_index_map_.insert(std::make_pair(name, i));
  }

  std::string ov_device_type = "CPU";
  ov::AnyMap properties;
  status = OpenVinoConvert::convertFromInferenceParam(
      openvino_inference_param, ov_device_type, properties);

  compiled_model_ = core_.compile_model(model, ov_device_type, properties);
  infer_request_ = compiled_model_.create_infer_request();

  for (auto iter : input_index_map_) {
    ov::Tensor ov_tensor = infer_request_.get_tensor(iter.first);
    device::Tensor *tensor = OpenVinoConvert::convertToTensor(ov_tensor);
    input_tensors_.insert(std::make_pair(iter.first, tensor));
  }
  for (auto iter : output_index_map_) {
    ov::Tensor ov_tensor = infer_request_.get_tensor(iter.first);
    device::Tensor *tensor = OpenVinoConvert::convertToTensor(ov_tensor);
    output_tensors_.insert(std::make_pair(iter.first, tensor));
  }

  return status;
}

base::Status OpenVinoInference::run() {
  base::Status status = base::kStatusCodeOk;
  device::Device *device = device::getDevice(inference_param_->device_type_);
  // inputs
  for (auto iter : external_input_tensors_) {
    device::Tensor *external_tensor = iter.second;
    ov::Tensor ov_tensor = OpenVinoConvert::convertFromTensor(external_tensor);
    infer_request_.set_tensor(iter.first, ov_tensor);
  }
  // outputs
  // for (auto iter : external_output_tensors_) {
  //   device::Tensor *external_tensor = iter.second;
  //   if (!external_tensor->empty()) {
  //     ov::Tensor ov_tensor =
  //         OpenVinoConvert::convertFromTensor(external_tensor);
  //     infer_request_.set_tensor(iter.first, ov_tensor);
  //   }
  // }
  // forward
  infer_request_.start_async();
  infer_request_.wait();
  // for (auto iter : external_output_tensors_) {
  //   device::Tensor *external_tensor = iter.second;
  //   if (external_tensor->empty()) {
  //     ov::Tensor ov_tensor = infer_request_.get_tensor(iter.first);
  //     OpenVinoConvert::convertToTensor(ov_tensor, external_tensor);
  //   }
  // }
  return status;
}

device::Tensor *OpenVinoInference::getOutputTensorAfterRun(
    const std::string &name, base::DeviceType device_type, bool is_copy,
    base::DataFormat data_format) {
  device::Device *device = device::getDevice(device_type);
  device::Tensor *internal_tensor = output_tensors_[name];
  device::TensorDesc desc = internal_tensor->getDesc();
  bool flag = is_copy || (internal_tensor->getDevice() != device);
  if (flag) {
    device::Tensor *output_tensor = new device::Tensor(device, desc, name);
    deepCopyBuffer(internal_tensor->getBuffer(), output_tensor->getBuffer());
    return output_tensor;
  } else {
    device::Tensor *output_tensor =
        new device::Tensor(desc, internal_tensor->getBuffer(), name);
    return output_tensor;
  }
}

}  // namespace inference
}  // namespace nndeploy
