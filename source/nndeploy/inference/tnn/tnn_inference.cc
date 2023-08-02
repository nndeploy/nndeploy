#include "nndeploy/inference/tnn/tnn_inference.h"

// TODO:
//      TNN的input_mat和blob有什么差别？
//      Init有一大堆待办

// Question:ststus和TNN的status是一个status吗？-不是，TNN的status是一个类，里面装了很多参数
// -可能需要两套status？ Question:Init里面应该不用管forward的事吧？ -不用，有run
// interpreter问题 -TNN的TNNImplRknpu::Init里面会创建，应该不用管这个
// min_shape在inference_param里面，取出来换成TNN的格式 -不用换，格式确认一致
// 模型路径问题，is_path； -这一层不处理

namespace nndeploy {
namespace inference {

/**
 *
 */
TypeInferenceRegister<TypeInferenceCreator<
    TnnInference>>  //<TypeInferenceCreator<TnnInference>>这个是参数T，TypeInferenceRegister这个参数是类
    g_tnn_inference_register(
        base::
            kInferenceTypeTnn);  // 用base::kInferenceTypeTnn去构造了一个g_tnn_inference_register

TnnInference::TnnInference(base::InferenceType type) : Inference(type) {
  tnn_ = nullptr;
  instance_ = nullptr;
}

TnnInference::~TnnInference() {}

base::Status TnnInference::init() {
  base::Status status = base::kStatusCodeOk;
  TnnInferenceParam *tnn_inference_param =
      dynamic_cast<TnnInferenceParam *>(inference_param_);

  if (device::isHostDeviceType(inference_param_->device_type_)) {
    is_share_command_queue_ = true;
  } else {
    is_share_command_queue_ = false;
  }

  status = TnnConvert::convertFromInferenceParam(
      tnn_inference_param, model_config_, network_config_);
  NNDEPLOY_RETURN_ON_NEQ(status, base::kStatusCodeOk, "init failed");

  tnn_ = new tnn::TNN();
  tnn::Status tnnststus = tnn_->Init(model_config_);
  if (tnnststus != tnn::StatusCode::TNN_OK) {
    NNDEPLOY_LOGE("TNN init failed!\n");
    return base::kStatusCodeErrorInferenceTnn;
  }

  is_input_dynamic_ = inference_param_->is_dynamic_shape_;
  is_output_dynamic_ = is_input_dynamic_;
  can_op_input_ = false;
  can_op_output_ = false;

  if (is_input_dynamic_) {
    instance_ = tnn_->CreateInst(network_config_, tnnstatus,
                                 inference_param_->min_shape_,
                                 inference_param_->max_shape_);  // shape
  } else {
    instance_ =
        tnn_->CreateInst(network_config_, tnnstatus);  // input_shape默认？
  }

  if (external_input_tensors_.isEmpty() == true) {
    tnn::BlobMap input_blobs;
    instance_->GetAllInputBlobs(input_blobs);
    for (auto iter : input_blobs) {
      input_tensors_ = TnnConvert::BlobConvertToTensor(
          iter->first, iter->second);  // 输入是external路径输出是不是一定也是？
    }
    tnn::BlobMap output_blobs;
    instance_->GetAllOutputBlobs(output_blobs);
    for (auto iter : output_blobs) {
      output_tensors_ =
          TnnConvert::BlobConvertToTensor(iter->first, iter->second);
    }
  } else {  // 这里和run里面重复了，记得一定要改！！！
    std::shared_ptr<tnn::Mat> input_mat =
        MatConvertFromTensor(external_input_tensors_);
    tnn::MatConvertParam param;
    instance_->SetInputMat(input_mat, param, input);

    external_output_tensors_ =
        MatConvertToTensor(instance_->GetOutputMat());  // 这里可能有错误
  }

  if (tnnststus != tnn::StatusCode::TNN_OK) {
    NNDEPLOY_LOGE("TNN init failed!\n");
    return base::kStatusCodeErrorInferenceTnn;
  }

  return status;
}

base::Status TnnInference::deinit() {
  deallocateInputOutputTensor();

  instance_->DeInit();
  tnn_->DeInit();
  delete tnn_;

  return base::kStatusCodeOk;
}

int64_t TnnInference::getMemorySize() {
  int memory_size;
  tnn::Status tnnststus = instance_->GetForwardMemorySize(memory_size);
  if (tnnststus != tnn::StatusCode::TNN_OK) {
    NNDEPLOY_LOGE("TNN GetForwardMemorySize failed!\n");
    return base::kStatusCodeErrorInferenceTnn;
  }
  return memory_size;
}

base::Status TnnInference::setMemory(device::Buffer *buffer) {
  int memory_size = getMemorySize();
  int memory_size_free = buffer->getSize();

  if (memory_size_free >= memory_size) {
    tnn::Status tnnststus = instance_->SetForwardMemory(buffer.data_ptr_);
    if (tnnststus != tnn::StatusCode::TNN_OK) {
      NNDEPLOY_LOGE("TNN SetForwardMemory failed!\n");
      return base::kStatusCodeErrorInferenceTnn;
    }
  } else {
    NNDEPLOY_LOGE("Buffer out of memory!\n");
    return base::kStatusCodeErrorInferenceTnn;
  }
  return base::kStatusCodeOk;
}

device::TensorDesc TnnInference::getInputTensorAlignDesc(
    const std::string &name) {
  // 只针对extern路径，data_format从TNN中get出来，可能操作不了，改个可操作的格式，最后用来创建input_mat
  // -什么叫操作不了的格式？从哪里get？
  if (external_input_tensors_.isEmpty() != true) {
    if (tnn_inference_param->inputs_data_format_ ==) {
    }
  }

  return base::kStatusCodeOk;
}

device::TensorDesc TnnInference::getOutputTensorAlignDesc(
    const std::string &name) {
  if (external_output_tensors_.isEmpty() != true) {
    if (tnn_inference_param->outputs_data_format_ ==) {
    }
  }

  return base::kStatusCodeOk;
}

base::Status TnnInference::run() {
  /*
  if (external_input_tensors_.isEmpty() != true) {
    std::shared_ptr<tnn::Mat> input_mat =
        MatConvertFromTensor(external_input_tensors_);
    tnn::MatConvertParam param;
    instance_->SetInputMat(input_mat, param, input);
  }
  */
  tnn::Status tnnststus = instance_->Forward();
  if (tnnststus != tnn::StatusCode::TNN_OK) {
    NNDEPLOY_LOGE("TNN forward failed!\n");
    return base::kStatusCodeErrorInferenceTnn;
  }

  return base::kStatusCodeOk;
}

}  // namespace inference
}  // namespace nndeploy