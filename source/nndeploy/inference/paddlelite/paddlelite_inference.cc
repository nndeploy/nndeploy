
#include "nndeploy/base/shape.h"
#include "nndeploy/inference/paddlelite/paddlelite_inference.h"
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
  PaddleLiteInferenceParam *paddlelite_inference_param =
      dynamic_cast<PaddleLiteInferenceParam *>(inference_param_);
  std::string model_buffer;
  if (paddlelite_inference_param->is_path_) {
    model_buffer = base::openFile(paddlelite_inference_param->model_value_[0]);
  } else {
    model_buffer = paddlelite_inference_param->model_value_[0];
  }
  
  config_.set_model_from_file(model_buffer);
  config_.set_power_mode(static_cast<paddle::lite_api::PowerMode>(power_mode_));
  config_.set_threads(paddlelite_inference_param->num_thread_);
  predictor_ = paddle::lite_api::CreatePaddlePredictor<paddle::lite_api::MobileConfig>(config_);

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

  return status;
}

}  // namespace inference
}  // namespace nndeploy
