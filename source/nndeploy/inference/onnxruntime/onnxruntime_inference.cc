
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

  std::string model_buffer;
  OnnxRuntimeInferenceParam *onnxruntime_inference_param =
      dynamic_cast<OnnxRuntimeInferenceParam *>(inference_param_);
  if (onnxruntime_inference_param->is_path_) {
    model_buffer = base::openFile(onnxruntime_inference_param->model_value_[0]);
  } else {
    model_buffer = onnxruntime_inference_param->model_value_[0];
  }

  OnnxRuntimeConvert::convertFromInferenceParam(onnxruntime_inference_param,
                                                &session_options_);
  session_ = {env_, onnx.data(), onnx.size(), session_options_};

  auto memory_info =
      Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
  Ort::Allocator allocator(session_, memory_info);

  auto n_inputs = session_.GetInputCount();
  for (int i = 0; i < n_inputs; ++i) {
#if ORT_API_VERSION >= 13
    auto input_name = session_.GetInputNameAllocated(i, allocator).release();
#else
    auto input_name = session_.GetInputName(i, allocator);
#endif
    auto type_info = session_.GetInputTypeInfo(i);
    auto shape = OnnxRuntimeConvert::convertToShape(
        type_info, onnxruntime_inference_param->is_dynamic_shape_,
        onnxruntime_inference_param->max_shape_);
    auto data_type = OnnxRuntimeConvert::convertToDataType(
        type_info.GetTensorTypeAndShapeInfo().GetElementType());
    input_tensors_.emplace_back(
        TensorDesc{device_, data_type, shape, input_name});
    allocator.Free(input_name);
  }

  auto n_outputs = session_.GetOutputCount();
  for (int i = 0; i < n_outputs; ++i) {
#if ORT_API_VERSION >= 13
    auto output_name = session_.GetOutputNameAllocated(i, allocator).release();
#else
    auto output_name = session_.GetOutputName(i, allocator);
#endif
    auto type_info = session_.GetOutputTypeInfo(i);
    auto shape = to_shape(type_info);
    NNDEPLOY_LOGI("output {}, shape = {}", i, shape);
    filter_shape(shape);
    OUTCOME_TRY(auto data_type,
                ConvertElementType(
                    type_info.GetTensorTypeAndShapeInfo().GetElementType()));
    output_tensors_.emplace_back(
        TensorDesc{device_, data_type, shape, output_name});
    allocator.Free(output_name);
  }

  is_share_command_queue_ = true;
  is_batch_ = (onnxruntime_inference_param->max_batch_size_ > 1);
  is_input_dynamic_ = inference_param_->is_dynamic_shape_;
  // TODO: 有可能输入非动态，但是输出是动态的
  is_output_dynamic_ = is_input_dynamic_;
  can_op_input_ = true;
  can_op_output_ = true;

  return status;
}

base::Status OnnxRuntimeInference::deinit() {
  base::Status status = base::kStatusCodeOk;
  for (auto iter : input_tensors_) {
    delete iter.second;
  }
  input_tensors_.clear();
  for (auto iter : output_tensors_) {
    delete iter.second;
  }
  output_tensors_.clear();
  return status;
}

base::Status OnnxRuntimeInference::reshape(base::ShapeMap &shape_map) {
  base::ShapeMap current_shape;
  auto n_inputs = session_.GetInputCount();
  for (int i = 0; i < n_inputs; ++i) {
    // input_name会被释放,是否需要拷贝一份呢?
#if ORT_API_VERSION >= 13
    auto input_name = session_.GetInputNameAllocated(i, allocator).release();
#else
    auto input_name = session_.GetInputName(i, allocator);
#endif
    auto type_info = session_.GetInputTypeInfo(i);
    auto shape = OnnxRuntimeConvert::convertToShape(type_info);
    OnnxRuntimeConvert::filterShape(shape);
    current_shape.insert({input_name, shape});
    allocator.Free(input_name);
  }
  bool flag = false;
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
  device::Device *device = device::getDevice(inference_param_->device_type_);
  try {
    OUTCOME_TRY(stream_.Wait());
    Ort::IoBinding binding(session_);
    std::vector<Ort::Value> inputs;
    std::vector<Ort::Value> outputs;
    Ort::RunOptions options;

    inputs.reserve(input_tensors_.size());
    for (auto &t : input_tensors_) {
      inputs.push_back(AsOrtValue(t));
      binding.BindInput(t.name(), inputs.back());
    }

    // TODO: We are in the same situation as PPL.nn, the backend can't infer
    // shapes
    //  without executing forward
    for (auto &t : output_tensors_) {
      binding.BindOutput(t.name(), MemoryInfo(t.desc()));
    }

    session_.Run({}, binding);

    outputs = binding.GetOutputValues();
    for (size_t i = 0; i < output_tensors_.size(); ++i) {
      OUTCOME_TRY(auto tmp, AsTensor(outputs[i], output_tensors_[i].device()));
      output_tensors_[i].Reshape(tmp.shape());
      OUTCOME_TRY(tmp.CopyTo(output_tensors_[i], stream_));
    }

    OUTCOME_TRY(stream_.Wait());
  } catch (const std::exception &e) {
    MMDEPLOY_ERROR(e.what());
    return Status(eFail);
  }
  return status;
}

}  // namespace inference
}  // namespace nndeploy
