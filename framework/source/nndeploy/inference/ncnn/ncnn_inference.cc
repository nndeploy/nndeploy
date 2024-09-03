
#include "nndeploy/inference/ncnn/ncnn_inference.h"

namespace nndeploy {
namespace inference {

TypeInferenceRegister<TypeInferenceCreator<NcnnInference>>
    g_ncnn_inference_register(base::kInferenceTypeNcnn);

NcnnInference::NcnnInference(base::InferenceType type) : Inference(type) {}
NcnnInference::~NcnnInference() {}

base::Status NcnnInference::init() {
  base::Status status = base::kStatusCodeOk;

  NcnnInferenceParam *ncnn_inference_param =
      dynamic_cast<NcnnInferenceParam *>(inference_param_);
  if (device::isHostDeviceType(inference_param_->device_type_)) {
    is_share_command_queue_ = true;
  } else {
    is_share_command_queue_ = false;
  }

  status =
      NcnnConvert::convertFromInferenceParam(ncnn_inference_param, net_.opt);
  NNDEPLOY_RETURN_ON_NEQ(status, base::kStatusCodeOk,
                         "convertFromInferenceParam failed");
  std::string params = ncnn_inference_param->model_value_[0];
  std::string weights = ncnn_inference_param->model_value_[1];
  if (ncnn_inference_param->model_type_ == base::kModelTypeNcnn) {
    if (ncnn_inference_param->is_path_) {
      int ret = net_.load_param(reinterpret_cast<const char *>(params.data()));
      if (ret != 0) {
        NNDEPLOY_LOGE("load_param failed! ret=%d\n", ret);
        return base::kStatusCodeErrorInferenceNcnn;
      }
      ret = net_.load_model(reinterpret_cast<const char *>(weights.data()));
      if (ret != 0) {
        NNDEPLOY_LOGE("load_model failed! ret=%d\n", ret);
        return base::kStatusCodeErrorInferenceNcnn;
      }
    } else {
      int ret = net_.load_param(
          reinterpret_cast<const unsigned char *>(params.data()));
      if (ret != 0) {
        NNDEPLOY_LOGE("load_param failed! ret=%d\n", ret);
        return base::kStatusCodeErrorInferenceNcnn;
      }
      ret = net_.load_model(
          reinterpret_cast<const unsigned char *>(weights.data()));
      if (ret != 0) {
        NNDEPLOY_LOGE("load_model failed! ret=%d\n", ret);
        return base::kStatusCodeErrorInferenceNcnn;
      }
    }
  } else {
    NNDEPLOY_LOGE("Not support model type[%d]!\n",
                  ncnn_inference_param->model_type_);
    return base::kStatusCodeErrorInferenceNcnn;
  }

  status = allocateInputOutputTensor();
  NNDEPLOY_RETURN_ON_NEQ(status, base::kStatusCodeOk,
                         "allocateInputOutputTensor failed!!\n");

  return status;
}

base::Status NcnnInference::deinit() {
  internal_outputs_.clear();
  base::Status status = deallocateInputOutputTensor();
  NNDEPLOY_RETURN_ON_NEQ(status, base::kStatusCodeOk,
                         "deallocateInputOutputTensor failed!!\n");
  return base::kStatusCodeOk;
}

base::Status NcnnInference::reshape(base::ShapeMap &shape_map) {
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
      return base::kStatusCodeErrorInferenceNcnn;
    }
  }

  if (flag) {
    for (auto iter : input_tensors_) {
      iter.second->deallocate();
      device::TensorDesc desc = iter.second->getDesc();
      desc.shape_.clear();
      desc.stride_.clear();
      desc.shape_ = shape_map[iter.first];
      iter.second->justModify(desc);
    }
    for (auto iter : output_tensors_) {
      iter.second->deallocate();
      device::TensorDesc desc = iter.second->getDesc();
      desc.shape_.clear();
      desc.stride_.clear();
      iter.second->justModify(desc);
    }
  }

  return base::kStatusCodeOk;
}

base::Status NcnnInference::run() {
  auto extractor = net_.create_extractor();
  for (auto iter : external_input_tensors_) {
    ncnn::Mat input_mat = NcnnConvert::matConvertFromTensor(iter.second);
    extractor.input(iter.first.c_str(), input_mat);
  }
  internal_outputs_.clear();
  for (auto iter : output_tensors_) {
    std::string output_name = iter.first;
    // TODO：这个内存是浅拷贝，需要保证这个内存在在这个函数结束之前不会被释放？
    ncnn::Mat output_mat;
    extractor.extract(output_name.c_str(), output_mat);
    internal_outputs_.insert({output_name, output_mat});
    device::Tensor *output_tensor = iter.second;
    NcnnConvert::matConvertToTensor(output_mat, output_name,
                                    output_tensor);  // 浅拷贝
  }
  return base::kStatusCodeOk;
}

device::Tensor *NcnnInference::getOutputTensorAfterRun(
    const std::string &name, base::DeviceType device_type, bool is_copy,
    base::DataFormat data_format) {
  device::Device *device = device::getDevice(device_type);
  device::Tensor *internal_tensor = output_tensors_[name];
  device::TensorDesc desc = internal_tensor->getDesc();
  bool flag = is_copy || (internal_tensor->getDevice() != device);
  if (flag) {
    device::Tensor *output_tensor = new device::Tensor(device, desc, name);
    internal_tensor->getBuffer()->copyTo(output_tensor->getBuffer());
    return output_tensor;
  } else {
    device::Tensor *output_tensor =
        new device::Tensor(desc, internal_tensor->getBuffer(), name);
    return output_tensor;
  }
}

base::Status NcnnInference::allocateInputOutputTensor() {
  const std::vector<ncnn::Blob> &blobs = net_.blobs();
  const std::vector<int> &input_indexes = net_.input_indexes();
  for (int i = 0; i < input_indexes.size(); i++) {
    ncnn::Blob blob = blobs[input_indexes[i]];
    // device::Tensor *input_tensor = NcnnConvert::blobConvertToTensor(blob);
    device::Tensor *input_tensor = new device::Tensor(blob.name);
    input_tensors_.insert({blob.name, input_tensor});
  }
  const std::vector<int> &output_indexes = net_.output_indexes();
  for (int i = 0; i < output_indexes.size(); i++) {
    ncnn::Blob blob = blobs[output_indexes[i]];
    // device::Tensor *output_tensor = NcnnConvert::blobConvertToTensor(blob);
    device::Tensor *output_tensor = new device::Tensor(blob.name);
    output_tensors_.insert({blob.name, output_tensor});
  }
  return base::kStatusCodeOk;
}

base::Status NcnnInference::deallocateInputOutputTensor() {
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