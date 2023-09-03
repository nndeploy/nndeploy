
#include "nndeploy/inference/ncnn/ncnn_inference.h"

namespace nndeploy {
namespace inference {

TypeInferenceRegister<TypeInferenceCreator<NcnnInference>>
    g_ncnn_inference_register(base::kInferenceTypeNcnn);

NcnnInference::NcnnInference(base::InferenceType type) : Inference(type) {
  net_ = nullptr;
}
NcnnInference::~NcnnInference() {}

// 默认为init之前inference_param已经初始化，利用inference_param初始化NCNN的instance和其余各项参数
base::Status NcnnInference::init() {
  base::Status status = base::kStatusCodeOk;

  net_ = new ncnn::Net();

  NcnnInferenceParam *ncnn_inference_param =
      dynamic_cast<NcnnInferenceParam *>(inference_param_);
  if (device::isHostDeviceType(inference_param_->device_type_)) {
    is_share_command_queue_ = true;
  } else {
    is_share_command_queue_ = false;
  }

  status =
      NcnnConvert::convertFromInferenceParam(ncnn_inference_param, net_->opt);
  NNDEPLOY_RETURN_ON_NEQ(status, base::kStatusCodeOk,
                         "convertFromInferenceParam failed");
  std::string params = ncnn_inference_param->model_value_[0];
  std::string weights = ncnn_inference_param->model_value_[1];
  if (ncnn_inference_param->model_type_ == base::kModelTypeNcnn) {
    if (ncnn_inference_param->is_path_) {
      net_->load_param(
          reinterpret_cast<const char *>((const unsigned char *)params.data()));
      net_->load_model(reinterpret_cast<const unsigned char *>(
          (const char *)weights.data()));
    } else {
      net_->load_param(reinterpret_cast<const unsigned char *>(
          (const unsigned char *)params.data()));
      net_->load_model(reinterpret_cast<const unsigned char *>(
          (const unsigned char *)weights.data()));
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
  base::Status status = deallocateInputOutputTensor();
  NNDEPLOY_RETURN_ON_NEQ(status, base::kStatusCodeOk,
                         "deallocateInputOutputTensor failed!!\n");
  delete net_;
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
      iter.second->deallocateBuffer();
      device::TensorDesc desc = iter.second->getDesc();
      desc.shape_.clear();
      desc.stride_.clear();
      desc.shape_ = shape_map[iter.first];
      iter.second->justModify(desc);
    }
    for (auto iter : output_tensors_) {
      iter.second->deallocateBuffer();
      device::TensorDesc desc = iter.second->getDesc();
      desc.shape_.clear();
      desc.stride_.clear();
      iter.second->justModify(desc);
    }
  }

  return base::kStatusCodeOk;
}

base::Status NcnnInference::run() {
  ncnn::Extractor extractor = net_->create_extractor();
  for (auto iter : external_input_tensors_) {
    ncnn::Mat input_mat = NcnnConvert::matConvertFromTensor(iter.second);
    extractor.input(iter.first.c_str(), input_mat);
  }
  for (auto iter : external_output_tensors_) {
    std::string output_name = iter.first;
    ncnn::Mat output_mat;
    extractor.extract(output_name.c_str(), output_mat);
    device::Tensor *output_tensor = iter.second;
    NcnnConvert::matConvertToTensor(output_mat, output_name,
                                    output_tensor);  // 浅拷贝
  }
  return base::kStatusCodeOk;
}

base::Status NcnnInference::allocateInputOutputTensor() {
  // ncnn::Extractor extractor = net_->create_extractor();
  const std::vector<ncnn::Blob> &blobs = net_->blobs();
  const std::vector<int> &input_indexes = net_->input_indexes();
  for (int i = 0; i < input_indexes.size(); i++) {
    ncnn::Blob blob = blobs[input_indexes[i]];
    device::Tensor *input_tensor = NcnnConvert::blobConvertToTensor(blob);
    input_tensors_.insert({blob.name, input_tensor});
  }
  const std::vector<int> &output_indexes = net_->output_indexes();
  for (int i = 0; i < output_indexes.size(); i++) {
    ncnn::Blob blob = blobs[output_indexes[i]];
    device::Tensor *output_tensor = NcnnConvert::blobConvertToTensor(blob);
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