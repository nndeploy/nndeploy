
#include "nndeploy/inference/mdc/mdc_inference.h"

#include "nndeploy/base/shape.h"
#include "nndeploy/inference/mdc/mdc_convert.h"
#include "nndeploy/inference/mdc/mdc_inference_param.h"

namespace nndeploy {
namespace inference {

TypeInferenceRegister<TypeInferenceCreator<MdcInference>> g_mdc_inference_register(base::kInferenceTypeMdc);

MdcInference::MdcInference(base::InferenceType type) : Inference(type) {}

MdcInference::~MdcInference() {}

base::Status MdcInference::init() {
  base::Status status = base::kStatusCodeOk;
  // is_share_command_queue_ = true;
  MdcInferenceParam *mdc_inference_param = dynamic_cast<MdcInferenceParam *>(inference_param_);

  device::Device *device = nullptr;

  if (mdc_inference_param->model_type_ == base::kModelTypeMdc) {
    aclError ret = aclInit(aclConfigPath);
    if (ret != ACL_SUCCESS) {
      NNDEPLOY_LOGE("aclInit failed, errorCode is %d", ret);
      status = base::kStatusCodeErrorInferenceMdc;
    }
    device = device::getDevice(inference_param_->device_type_);
    context_ = device->getCommandQueue();

    if (mdc_inference_param->is_path_) {
      aclError ret = aclmdlLoadFromFile(mdc_inference_param->model_value_[0].c_str(), &modelId_);
      if (ret != ACL_SUCCESS) {
        NNDEPLOY_LOGE("aclmdlLoadFromFile failed, errorCode is %d", ret);
        status = base::kStatusCodeErrorInferenceMdc;
      }
    } else {
      NNDEPLOY_LOGE("Currently, direct input is not supported. Please use the om file as input");
      status = base::kStatusCodeErrorInferenceMdc;
    }

    modelDesc_ = aclmdlCreateDesc();
    ret = aclmdlGetDesc(modelDesc_, modelId_);
    if (ret != ACL_SUCCESS) {
      NNDEPLOY_LOGE("aclmdlGetDesc failed, errorCode is %d", ret);
      status = base::kStatusCodeErrorInferenceMdc;
    }
    NNDEPLOY_RETURN_ON_NEQ(status, base::kStatusCodeOk, "initWithMdcModel failed");
  } else {
    NNDEPLOY_LOGE("not support this model type(%d)!\n", mdc_inference_param->model_type_);
    return base::kStatusCodeErrorInferenceMdc;
  }

  size_t n_inputs = aclmdlGetNumInputs(modelDesc_);
  for (auto i = 0; i < n_inputs; ++i) {
    std::string input_name = std::string(aclmdlGetInputNameByIndex(modelDesc_, i));
    std::vector<int64_t> input_shape;
    aclmdlIODims input_dim;
    aclError ret = aclmdlGetInputDims(modelDesc_, i, &input_dim);
    if (ret != ACL_SUCCESS) {
      NNDEPLOY_LOGE("init: Get input_{%d} dims failed, errorCode is %d", i, ret);
      status = base::kStatusCodeErrorInferenceMdc;
    }
    for (int dim_index = 0; dim_index < input_dim.dimCount; dim_index++)
      input_shape.push_back(input_dim.dims[dim_index]);
    aclDataType data_type = aclmdlGetInputDataType(modelDesc_, i);
    inputs_desc_.emplace_back(OrtValueInfo{input_name, input_shape, data_type});

    device::TensorDesc desc;
    if (mdc_inference_param->max_shape_.find(input_name) != mdc_inference_param->max_shape_.end()) {
      desc.shape_ = MdcConvert::convertToShape(input_shape, mdc_inference_param->max_shape_[input_name]);
    } else {
      desc.shape_ = MdcConvert::convertToShape(input_shape);
    }
    desc.data_type_ = MdcConvert::convertToDataType(data_type);
    desc.data_format_ = MdcConvert::getDataFormatByShape(desc.shape_);

    device::Tensor *max_input_tensor = new device::Tensor(device, desc, input_name);
    max_input_tensors_.insert({input_name, max_input_tensor});

    device::Buffer *max_input_buffer = max_input_tensor->getBuffer();
    device::Tensor *current_input_tensor = new device::Tensor(desc, max_input_buffer, input_name);
    input_tensors_.insert({input_name, current_input_tensor});
  }

  for (auto iter : input_tensors_) {
    batch_size_ = iter.second->getBatch();
  }

  size_t n_outputs = aclmdlGetNumOutputs(modelDesc_);
  for (auto i = 0; i < n_outputs; ++i) {
    std::string output_src_name = std::string(aclmdlGetOutputNameByIndex(modelDesc_, i));
    // 由于mdc输出名称会自动加前缀且用:分开，这里找出真正的输出名称，对齐onnxruntime
    std::string output_name = base::split_string(output_src_name, ":").back();
    mdc_change_output_names_.insert({output_src_name, output_name});
    std::vector<int64_t> output_shape;
    aclmdlIODims output_dim;
    aclError ret = aclmdlGetOutputDims(modelDesc_, i, &output_dim);
    if (ret != ACL_SUCCESS) {
      NNDEPLOY_LOGE("init: Get output_{%d} shape failed, errorCode is %d", i, ret);
      status = base::kStatusCodeErrorInferenceMdc;
    }
    for (int dim_index = 0; dim_index < output_dim.dimCount; dim_index++)
      output_shape.push_back(output_dim.dims[dim_index]);
    aclDataType data_type = aclmdlGetOutputDataType(modelDesc_, i);
    outputs_desc_.emplace_back(OrtValueInfo{output_name, output_shape, data_type});

    if (!isDynamic(output_shape)) {
      device::TensorDesc desc;
      desc.shape_ = MdcConvert::convertToShape(output_shape);
      desc.shape_[0] = batch_size_;
      desc.data_type_ = MdcConvert::convertToDataType(data_type);
      desc.data_format_ = MdcConvert::getDataFormatByShape(desc.shape_);
      device::Tensor *max_output_tensor = new device::Tensor(device, desc, output_name);
      max_output_tensors_.insert({output_name, max_output_tensor});

      device::Buffer *max_output_buffer = max_output_tensor->getBuffer();
      device::Tensor *current_output_tensor = new device::Tensor(desc, max_output_buffer, output_name);
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

base::Status MdcInference::deinit() {
  base::Status status = base::kStatusCodeOk;
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
  ReleaseAllResource();
  return status;
}

base::Status MdcInference::reshape(base::ShapeMap &shape_map) {
  base::Status status = base::kStatusCodeOk;

  base::ShapeMap current_shape;
  size_t n_inputs = aclmdlGetNumInputs(modelDesc_);
  for (auto i = 0; i < n_inputs; ++i) {
    std::string input_name = std::string(aclmdlGetInputNameByIndex(modelDesc_, i));
    std::vector<int64_t> input_shape;
    aclmdlIODims input_dim;
    aclError ret = aclmdlGetInputDims(modelDesc_, i, &input_dim);
    if (ret != ACL_SUCCESS) {
      NNDEPLOY_LOGE("reshape: Get input_{%d} shape failed, errorCode is %d", i, ret);
      status = base::kStatusCodeErrorInferenceMdc;
    }
    for (int dim_index = 0; dim_index < input_dim.dimCount; dim_index++)
      input_shape.push_back(input_dim.dims[dim_index]);
    auto shape = MdcConvert::convertToShape(input_shape);
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
      NNDEPLOY_LOGE("reshape failed, not found input tensor(%s)!\n", iter.first.c_str());
      return base::kStatusCodeErrorInferenceMdc;
    }
  }
  return status;
}

base::Status MdcInference::run() {
  base::Status status = base::kStatusCodeOk;

  inputDataset_ = aclmdlCreateDataset();
  outputDataset_ = aclmdlCreateDataset();

  aclError ret = aclrtSetCurrentContext(context_);
  if (ret != ACL_SUCCESS) {
    NNDEPLOY_LOGE("aclrtSetCurrentContext failed, errorCode is %d", ret);
    return base::kStatusCodeErrorInferenceMdc;
  }
  device::Device *device = device::getDevice(inference_param_->device_type_);

  try {
    size_t n_inputs = aclmdlGetNumInputs(modelDesc_);
    for (auto i = 0; i < n_inputs; ++i) {
      std::string input_name = std::string(aclmdlGetInputNameByIndex(modelDesc_, i));
      device::Tensor *external_tensor = external_input_tensors_[input_name];
      device::Buffer *extern_buffer = external_tensor->getBuffer();
      device::Tensor *internal_tensor = input_tensors_[input_name];
      device::Buffer *internal_buffer = internal_tensor->getBuffer();
      base::DeviceType device_type = extern_buffer->getDeviceType();
      if (device::isHostDeviceType(device_type)) {
        device->upload(extern_buffer, internal_buffer);
      } else if (device_type == device->getDeviceType()) {
        device->copy(extern_buffer, internal_buffer);
      } else {
        NNDEPLOY_LOGE("mdc run failed, device type is not supported!\n");
        return base::kStatusCodeErrorInferenceMdc;
      }

      device::Buffer *input_buffer = input_tensors_[input_name]->getBuffer();
      aclDataBuffer *inputData = aclCreateDataBuffer(input_buffer->getPtr(), input_buffer->getDesc().size_[0]);
      aclError ret = aclmdlAddDatasetBuffer(inputDataset_, inputData);
      if (ret != ACL_SUCCESS) {
        NNDEPLOY_LOGE("input_{%d}: aclmdlAddDatasetBuffer failed, errorCode is %d.", i, ret);
        return base::kStatusCodeErrorInferenceMdc;
      }
    }

    size_t n_outputs = aclmdlGetNumOutputs(modelDesc_);
    for (auto i = 0; i < n_outputs; ++i) {
      std::string output_src_name = std::string(aclmdlGetOutputNameByIndex(modelDesc_, i));
      device::Buffer *output_buffer = output_tensors_[mdc_change_output_names_[output_src_name]]->getBuffer();
      aclDataBuffer *outputData = aclCreateDataBuffer(output_buffer->getPtr(), output_buffer->getDesc().size_[0]);
      aclError ret = aclmdlAddDatasetBuffer(outputDataset_, outputData);
      if (ret != ACL_SUCCESS) {
        NNDEPLOY_LOGE("ouput_{%d}: aclmdlAddDatasetBuffer failed, errorCode is %d.", i, ret);
        return base::kStatusCodeErrorInferenceMdc;
      }
    }

    // forward
    aclError ret = aclmdlExecute(modelId_, inputDataset_, outputDataset_);
    if (ret != ACL_SUCCESS) {
      NNDEPLOY_LOGE("mdc execute model failed, errorCode is %d", ret);
      return base::kStatusCodeErrorInferenceMdc;
    }

    status = device->synchronize();
    NNDEPLOY_RETURN_ON_NEQ(status, base::kStatusCodeOk, "synchronize failed");

    // outputs
    // for (auto i = 0; i < n_outputs; ++i) {
    //   std::string output_src_name = std::string(aclmdlGetOutputNameByIndex(modelDesc_, i));
    //   device::Tensor *external_tensor = external_output_tensors_[mdc_change_output_names_[output_src_name]];
    //   device::Buffer *extern_buffer = external_tensor->getBuffer();
    //   device::Tensor *internal_tensor = output_tensors_[mdc_change_output_names_[output_src_name]];
    //   device::Buffer *internal_buffer = internal_tensor->getBuffer();
    //   base::DeviceType device_type = extern_buffer->getDeviceType();
    //   if (device::isHostDeviceType(device_type)) {
    //     device->download(internal_buffer, extern_buffer);
    //   } else if (device_type == device->getDeviceType()) {
    //     device->copy(internal_buffer, extern_buffer);
    //   } else {
    //     NNDEPLOY_LOGE("run failed, device type is not supported!\n");
    //     return base::kStatusCodeErrorInferenceMdc;
    //   }
    // }
  } catch (const std::exception &e) {
    NNDEPLOY_LOGE("%s.\n", e.what());
    status = base::kStatusCodeErrorInferenceMdc;
  }
  return status;
}

device::Tensor *MdcInference::getOutputTensorAfterRun(const std::string &name, base::DeviceType device_type,
                                                      bool is_copy, base::DataFormat data_format) {
  device::Device *device = device::getDevice(device_type);
  device::Tensor *internal_tensor = output_tensors_[name];
  device::TensorDesc desc = internal_tensor->getDesc();
  bool flag = is_copy || (internal_tensor->getDevice() != device);
  if (flag) {
    device::Tensor *output_tensor = new device::Tensor(device, desc, name);
    deepCopyBuffer(internal_tensor->getBuffer(), output_tensor->getBuffer());
    return output_tensor;
  } else {
    device::Tensor *output_tensor = new device::Tensor(desc, internal_tensor->getBuffer(), name);
    return output_tensor;
  }
}

bool MdcInference::isDynamic(std::vector<int64_t> &shape) {
  int size = shape.size();
  for (int i = 1; i < size; ++i) {
    if (shape[i] < 0) {
      return true;
    }
  }
  return false;
}

void MdcInference::ReleaseAllResource() {
  aclError ret;
  // release resource includes acl resource, data set and unload model
  for (size_t i = 0; i < aclmdlGetDatasetNumBuffers(inputDataset_); ++i) {
    aclDataBuffer *dataBuffer = aclmdlGetDatasetBuffer(inputDataset_, i);
    (void)aclDestroyDataBuffer(dataBuffer);
    dataBuffer = nullptr;
  }
  (void)aclmdlDestroyDataset(inputDataset_);
  inputDataset_ = nullptr;

  for (size_t i = 0; i < aclmdlGetDatasetNumBuffers(outputDataset_); ++i) {
    aclDataBuffer *dataBuffer = aclmdlGetDatasetBuffer(outputDataset_, i);
    void *data = aclGetDataBufferAddr(dataBuffer);
    (void)aclrtFree(data);
    (void)aclDestroyDataBuffer(dataBuffer);
    dataBuffer = nullptr;
  }
  (void)aclmdlDestroyDataset(outputDataset_);
  outputDataset_ = nullptr;

  ret = aclmdlDestroyDesc(modelDesc_);
  if (ret != ACL_SUCCESS) {
    NNDEPLOY_LOGE("destroy description failed, errorCode is %d", ret);
  }

  ret = aclmdlUnload(modelId_);
  if (ret != ACL_SUCCESS) {
    NNDEPLOY_LOGE("unload model failed, errorCode is %d", ret);
  }

  ret = aclFinalize();
  if (ret != ACL_SUCCESS) {
    NNDEPLOY_LOGE("aclFinalize failed, errorCode is %d", ret);
  }
}

}  // namespace inference
}  // namespace nndeploy
