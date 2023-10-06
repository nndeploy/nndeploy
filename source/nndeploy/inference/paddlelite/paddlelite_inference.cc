
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
    is_share_command_queue_ = true;
  } else {
    is_share_command_queue_ = false;
  }

  PaddleLiteInferenceParam *paddlelite_inference_param =
      dynamic_cast<PaddleLiteInferenceParam *>(inference_param_);
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
  base::Status status = base::kStatusCodeOk;
  return status;
}

base::Status PaddleLiteInference::reshape(base::ShapeMap &shape_map) {
  base::Status status = base::kStatusCodeOk;
  return status;
}

base::Status PaddleLiteInference::run() {
  base::Status status = base::kStatusCodeOk;
  // inputs
  for (auto iter : external_input_tensors_) {
  }
  // forward
  predictor_->Run();
  // outputs
  for (auto iter : external_output_tensors_) {
  }
  return status;
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
    // device::Tensor *tensor =
    //     PaddleLiteConvert::convertToTensor(pd_tensor, name, device);
    input_tensors_.insert({name, tensor});
  }
  std::vector<std::string> output_names = predictor_->GetOutputNames();
  for (size_t i = 0; i < output_names.size(); ++i) {
    io_name_index_[output_names[i]] = i;
    TensorInfo info;
    std::unique_ptr<paddle::lite_api::Tensor> pd_tensor =
        predictor_->GetOutput(i);
    // device::Tensor *tensor =
    //     PaddleLiteConvert::convertToTensor(pd_tensor, name, device);
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
