
#include "nndeploy/inference/paddlelite/paddlelite_inference.h"

#include "nndeploy/base/shape.h"
#include "nndeploy/inference/paddlelite/paddlelite_convert.h"
#include "nndeploy/inference/paddlelite/paddlelite_inference_param.h"

namespace nndeploy {
namespace inference {

TypeInferenceRegister<TypeInferenceCreator<PaddleLiteInference>>
    g_paddlelite_inference_register(base::kInferenceTypePaddleLite);

PaddleLiteInference::PaddleLiteInference(base::InferenceType type)
    : Inference(type) {}

PaddleLiteInference::~PaddleLiteInference() {}

base::Status PaddleLiteInference::init() {
  base::Status status = base::kStatusCodeOk;

  if (device::isHostDeviceType(inference_param_->device_type_)) {
    is_share_context_ = true;
  } else {
    is_share_context_ = false;
  }

  PaddleLiteInferenceParam *paddlelite_inference_param =
      dynamic_cast<PaddleLiteInferenceParam *>(inference_param_.get());
  PaddleLiteConvert::convertFromInferenceParam(paddlelite_inference_param,
                                               config_);
  predictor_ =
      paddle::lite_api::CreatePaddlePredictor<paddle::lite_api::CxxConfig>(
          config_);

  status = allocateInputOutputTensor();
  NNDEPLOY_RETURN_ON_NEQ(status, base::kStatusCodeOk,
                         "allocateInputOutputTensor failed!!\n");
  return status;
}

base::Status PaddleLiteInference::deinit() {
  base::Status status = deallocateInputOutputTensor();
  NNDEPLOY_RETURN_ON_NEQ(status, base::kStatusCodeOk,
                         "deallocateInputOutputTensor failed!!\n");
  return status;
}

base::Status PaddleLiteInference::reshape(base::ShapeMap &shape_map) {
  base::Status status = base::kStatusCodeOk;
  bool flag = false;
  for (auto iter : shape_map) {
    std::string name = iter.first;
    IntVector shape = iter.second;
    if (base::shapeEqual(shape, input_tensors_[name]->getShape())) {
      continue;
    }
    flag = true;
    auto io_iter = io_name_index_.find(name);
    if (io_iter == io_name_index_.end()) {
      NNDEPLOY_LOGE("Cannot find input with name: %s in loaded model.\n",
                    name.c_str());
      return base::kStatusCodeErrorInferencePaddleLite;
    }
    auto pd_tensor = predictor_->GetInput(io_iter->second);
    paddle::lite_api::shape_t pd_shape =
        PaddleLiteConvert::convertToShape(shape);
    pd_tensor->Resize(pd_shape);
  }
  if (flag) {
    status = deallocateInputOutputTensor();
    NNDEPLOY_RETURN_ON_NEQ(status, base::kStatusCodeOk,
                           "deallocateInputOutputTensor failed!!\n");
    status = allocateInputOutputTensor();
    NNDEPLOY_RETURN_ON_NEQ(status, base::kStatusCodeOk,
                           "allocateInputOutputTensor failed!!\n");
  }
  return status;
}

base::Status PaddleLiteInference::run() {
  base::Status status = base::kStatusCodeOk;
  // inputs
  for (auto iter : external_input_tensors_) {
    std::string name = iter.first;
    device::Tensor *tensor = iter.second;

    auto io_iter = io_name_index_.find(name);
    if (io_iter == io_name_index_.end()) {
      NNDEPLOY_LOGE("Cannot find input with name: %s in loaded model.\n",
                    name.c_str());
      return base::kStatusCodeErrorInferencePaddleLite;
    }
    // copy data to input tensor, 那这里只能时host端的数据了
    status = device::deepCopyTensor(tensor, input_tensors_[name]);
    NNDEPLOY_RETURN_ON_NEQ(status, base::kStatusCodeOk,
                           "device::deepCopyTensor!\n");
  }
  // forward
  predictor_->Run();
  return status;
}

device::Tensor *PaddleLiteInference::getOutputTensorAfterRun(
    const std::string &name, base::DeviceType device_type, bool is_copy,
    base::DataFormat data_format) {
  auto iter = io_name_index_.find(name);
  if (iter == io_name_index_.end()) {
    NNDEPLOY_LOGE("Cannot find output with name: %s in loaded model.\n",
                  name.c_str());
    return nullptr;
  }
  std::unique_ptr<paddle::lite_api::Tensor> pd_tensor =
      predictor_->GetOutput(iter->second);
  if (pd_tensor == nullptr) {
    NNDEPLOY_LOGE("predictor_->GetOutput failed.\n");
    return nullptr;
  }
  bool can_op_flag = true;
  can_op_flag = can_op_flag && is_share_context_;
  device::Device *device = device::getDefaultHostDevice();
  device::Tensor *internal_tensor =
      PaddleLiteConvert::convertToTensor(pd_tensor);
  if (is_copy || !can_op_flag) {
    device::TensorDesc desc = this->getInputTensorAlignDesc(name);
    device::Tensor *output_tensor = new device::Tensor(device, desc, name);
    device::deepCopyTensor(internal_tensor, output_tensor);
    return output_tensor;
  } else {
    device::Tensor *output_tensor =
        MnnConvert::convertToTensor(internal_tensor, name, device);
    return output_tensor;
  }
}

base::Status PaddleLiteInference::allocateInputOutputTensor() {
  device::Device *device = nullptr;
  if (device::isHostDeviceType(inference_param_->device_type_)) {
    device = device::getDevice(inference_param_->device_type_);
  }

  std::vector<std::string> input_names = predictor_->GetInputNames();
  for (size_t i = 0; i < input_names.size(); ++i) {
    io_name_index_[input_names[i]] = i;
    std::unique_ptr<paddle::lite_api::Tensor> pd_tensor =
        predictor_->GetInput(i);
    device::Tensor *tensor =
        PaddleLiteConvert::convertToTensor(pd_tensor, name, device);
    input_tensors_.insert({name, tensor});
  }
  std::vector<std::string> output_names = predictor_->GetOutputNames();
  for (size_t i = 0; i < output_names.size(); ++i) {
    io_name_index_[output_names[i]] = i;
    TensorInfo info;
    std::unique_ptr<paddle::lite_api::Tensor> pd_tensor =
        predictor_->GetOutput(i);
    device::Tensor *tensor =
        PaddleLiteConvert::convertToTensor(pd_tensor, name, device);
    output_tensors_.insert({name, tensor});
  }

  return base::kStatusCodeOk;
}

base::Status PaddleLiteInference::deallocateInputOutputTensor() {
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
