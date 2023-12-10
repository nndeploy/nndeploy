#include "nndeploy/inference/tnn/tnn_inference.h"

namespace nndeploy {
namespace inference {

TypeInferenceRegister<TypeInferenceCreator<TnnInference>>
    g_tnn_inference_register(base::kInferenceTypeTnn);

TnnInference::TnnInference(base::InferenceType type) : Inference(type) {
  tnn_ = nullptr;
  instance_ = nullptr;
}
TnnInference::~TnnInference() {}

// 默认为init之前inference_param已经初始化，利用inference_param初始化TNN的instance和其余各项参数
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
  NNDEPLOY_RETURN_ON_NEQ(status, base::kStatusCodeOk,
                         "convertFromInferenceParam failed");

  tnn_ = new tnn::TNN();
  tnn::Status tnnstatus = tnn_->Init(model_config_);
  if (tnnstatus != tnn::StatusCode::TNN_OK) {
    NNDEPLOY_LOGE("TNN init failed!\n");
    return base::kStatusCodeErrorInferenceTnn;
  }

  if (inference_param_->is_dynamic_shape_) {
    instance_ = tnn_->CreateInst(network_config_, tnnstatus,
                                 inference_param_->min_shape_,
                                 inference_param_->max_shape_);  // shape
  } else {
    instance_ =
        tnn_->CreateInst(network_config_, tnnstatus);  // input_shape默认？
  }
  if (tnnstatus != tnn::StatusCode::TNN_OK) {
    NNDEPLOY_LOGE("TNN init failed!\n");
    return base::kStatusCodeErrorInferenceTnn;
  }

  status = allocateInputOutputTensor();
  NNDEPLOY_RETURN_ON_NEQ(status, base::kStatusCodeOk,
                         "allocateInputOutputTensor failed!!\n");

  return status;
}

base::Status TnnInference::deinit() {
  base::Status status = deallocateInputOutputTensor();
  NNDEPLOY_RETURN_ON_NEQ(status, base::kStatusCodeOk,
                         "deallocateInputOutputTensor failed!!\n");

  instance_->DeInit();
  tnn_->DeInit();
  delete tnn_;

  return base::kStatusCodeOk;
}

// 获取运行需要的memory大小
int64_t TnnInference::getMemorySize() {
  int memory_size;
  tnn::Status tnnststus = instance_->GetForwardMemorySize(memory_size);
  if (tnnststus != tnn::StatusCode::TNN_OK) {
    NNDEPLOY_LOGE("TNN GetForwardMemorySize failed!\n");
    return base::kStatusCodeErrorInferenceTnn;
  }
  return (int64_t)memory_size;
}

// 输入空指针就行，不用自己使用getMemorySize函数
base::Status TnnInference::setMemory(device::Buffer *buffer) {
  int memory_size = getMemorySize();
  int memory_size_free = buffer->getSize();

  if (memory_size_free >= memory_size) {
    tnn::Status tnnstatus = instance_->SetForwardMemory(buffer->getPtr());  //
    if (tnnstatus != tnn::StatusCode::TNN_OK) {
      NNDEPLOY_LOGE("TNN SetForwardMemory failed!\n");
      return base::kStatusCodeErrorInferenceTnn;
    }
  } else {
    NNDEPLOY_LOGE("Buffer memory size[%d] < forward_memory_size[%d]!\n",
                  memory_size_free, memory_size);
    return base::kStatusCodeErrorInferenceTnn;
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
        flag = true;
      }
    } else {
      NNDEPLOY_LOGE("Not exsit input[%s].\n", iter.first.c_str());
      return base::kStatusCodeErrorInferenceTnn;
    }
  }

  if (flag) {
    instance_->Reshape(shape_map);
    deallocateInputOutputTensor();
    allocateInputOutputTensor();
  }

  return base::kStatusCodeOk;
}

base::Status TnnInference::run() {
  tnn::MatConvertParam param = tnn::MatConvertParam();
  for (auto iter : external_input_tensors_) {
    std::shared_ptr<tnn::Mat> input_mat(
        TnnConvert::matConvertFromTensor(iter.second));
    instance_->SetInputMat(input_mat, param, iter.first);
  }
  tnn::Status tnnststus = instance_->Forward();
  if (tnnststus != tnn::StatusCode::TNN_OK) {
    NNDEPLOY_LOGE("TNN forward failed!\n");
    return base::kStatusCodeErrorInferenceTnn;
  }
  // for (auto iter : external_output_tensors_) {
  //   std::string output_name = iter.first;
  //   base::DeviceType device_type = iter.second->getDeviceType();
  //   tnn::DeviceType tnn_device_type =
  //       TnnConvert::convertFromDeviceType(device_type);
  //   std::shared_ptr<tnn::Mat> mat;
  //   instance_->GetOutputMat(mat, param, iter.first, tnn_device_type);
  //   device::Tensor *tmp_tenosr =
  //       TnnConvert::matConvertToTensor(mat.get(), output_name);
  //   device::Device *device = device::getDevice(device_type);
  //   device->copy(tmp_tenosr->getBuffer(), iter.second->getBuffer());
  //   delete tmp_tenosr;
  // }

  return base::kStatusCodeOk;
}

device::Tensor *TnnInference::getOutputTensorAfterRun(
    const std::string &name, base::DeviceType device_type, bool is_copy,
    base::DataFormat data_format) {
  device::Device *device = device::getDevice(device_type);
  tnn::DeviceType tnn_device_type =
      TnnConvert::convertFromDeviceType(device_type);
  tnn::MatConvertParam param = tnn::MatConvertParam();
  std::shared_ptr<tnn::Mat> mat;
  instance_->GetOutputMat(mat, param, name, tnn_device_type);
  output_mat_map_[name] = mat;
  device::Tensor *internal_tensor =
      TnnConvert::matConvertToTensor(mat.get(), name);
  device::TensorDesc desc = internal_tensor->getDesc();
  bool flag = is_copy || (internal_tensor->getDevice() != device);
  device::Tensor *output_tensor = nullptr;
  if (flag) {
    output_tensor = new device::Tensor(device, desc, name);
    deepCopyBuffer(internal_tensor->getBuffer(), output_tensor->getBuffer());
    delete internal_tensor;
    return output_tensor;
  } else {
    output_tensor = internal_tensor;
    return output_tensor;
  }
}

base::Status TnnInference::allocateInputOutputTensor() {
  tnn::BlobMap input_blobs;
  instance_->GetAllInputBlobs(input_blobs);
  for (auto iter : input_blobs) {
    std::string name = iter.first;
    tnn::Blob *blob = iter.second;
    device::Tensor *input_tensor = TnnConvert::blobConvertToTensor(blob);
    input_tensors_.insert({name, input_tensor});
  }
  tnn::BlobMap output_blobs;
  instance_->GetAllOutputBlobs(output_blobs);
  for (auto iter : output_blobs) {
    std::string name = iter.first;
    tnn::Blob *blob = iter.second;
    device::Tensor *output_tensor = TnnConvert::blobConvertToTensor(blob);
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
  output_mat_map_.clear();
  return base::kStatusCodeOk;
}

}  // namespace inference
}  // namespace nndeploy