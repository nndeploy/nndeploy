
#include "nndeploy/source/inference/tensor_rt/tensor_rt_inference.h"

#include <errno.h>

#include <iterator>

#include "nndeploy/source/base/shape.h"
#include "nndeploy/source/inference/tensor_rt/tensor_rt_convert.h"
#include "nndeploy/source/inference/tensor_rt/tensor_rt_inference_param.h"

namespace nndeploy {
namespace inference {

TypeInferenceRegister<TypeInferenceCreator<TensorRtInference>>
    g_internal_inference_register(base::kInferenceTypeTensorRt);

TensorRtLogger TensorRtInference::logger_;

TensorRtInference::TensorRtInference() {
  engine_ = nullptr;
  context_ = nullptr;
}

TensorRtInference::~TensorRtInference() {}

base::Status TensorRtInference::init() {
  base::Status status = base::kStatusCodeOk;

  min_shape_ = inference_param->min_shape_;
  opt_shape_ = inference_param->opt_shape_;
  max_shape_ = inference_param->max_shape_;

  std::string model_buffer;
  TensorRtInferenceParam *tensor_rt_inference_param =
      dynamic_cast<TensorRtInferenceParam *>(inference_param_);
  if (tensor_rt_inference_param->is_path_) {
    model_buffer = base::openFile(tensor_rt_inference_param->model_value_[0]);
  } else {
    model_buffer = tensor_rt_inference_param->model_value_[0];
  }

  if (tensor_rt_inference_param->model_type_ ==
      base::kInferenceTypeOnnxRuntime) {
    status = preRunWithOnnxModel(model_buffer, tensor_rt_inference_param);
    NNDEPLOY_RETURN_ON_NEQ(status, base::kStatusCodeOk);
  } else if (tensor_rt_inference_param->model_type_ ==
             base::kInferenceTypeTensorRt) {
    status = preRunWithTensorRtModel(model_buffer, tensor_rt_inference_param);
    NNDEPLOY_RETURN_ON_NEQ(status, base::kStatusCodeOk);
  } else {
    return base::kStatusCodeErrorInferenceTensorRt;
  }

  auto num_binds = engine_->getNbBindings();
  for (auto i = 0; i < num_binds; ++i) {
    std::string name = std::string(engine_->getBindingName(i));
    auto shape =
        TensorRtConvert::convertToShape(engine_->getBindingDimensions(i));
    if (engine_->bindingIsInput(i)) {
      current_shape_.insert({name, shape});
    }
  }

  status = reshape(max_shape_);
  NNDEPLOY_RETURN_ON_NEQ(status, base::kStatusCodeOk);

  auto num_binds = engine_->getNbBindings();
  bindings_.resize(num_binds);
  for (auto i = 0; i < num_binds; ++i) {
    std::string name = std::string(engine_->getBindingName(i));
    base::IntVector shape =
        TensorRtConvert::convertToShape(engine_->getBindingDimensions(i));
    base::DataType data_type =
        TensorRtConvert::convertToDataType(engine_->getBindingDataType(i));
    base::DataType data_format =
        TensorRtConvert::convertToDataFormat(engine_->getBindingFormat(i));

    if (engine_->bindingIsInput(i)) {
      std::shared_ptr<device::Tensor> max_input_tensor;
      device::Device *device =
          device::getDevice(inference_param_->device_type_);
      device::TensorDesc desc;
      desc.data_type_ = data_type;
      desc.format_ = data_format;
      desc.shape_ = shape;
      max_input_tensor.create(device, desc, name);
      device::Buffer *max_input_buffer = max_input_tensor->getBuffer();
      std::shared_ptr<device::Tensor> current_input_tensor;
      current_input_tensor.reset(
          new device::Tensor(desc, max_input_buffer, name));
      current_input_tensors_.insert({name, current_input_tensor});

      bindings_[i] = max_input_buffer->getPtr();
    } else {
      std::shared_ptr<device::Tensor> max_output_tensor;
      device::Device *device =
          device::getDevice(inference_param_->device_type_);
      device::TensorDesc desc;
      desc.data_type_ = data_type;
      desc.format_ = data_format;
      desc.shape_ = shape;
      max_output_tensor.create(device, desc, name);
      device::Buffer *max_output_buffer = max_output_tensor->getBuffer();
      std::shared_ptr<device::Tensor> current_output_tensor;
      current_output_tensor.reset(
          new device::Tensor(desc, max_output_buffer, name));
      current_output_tensors_.insert({name, current_output_tensor});

      bindings_[i] = max_output_buffer->getPtr();
    }
    io_name_index_[name] = i;
  }

  return status;
}

base::Status TensorRtInference::deinit() {
  base::Status status = base::kStatusCodeOk;
  device::Device *device = device::getDevice(inference_param_->device_type_);
  if (inner_forward_buffer_ != nullptr) {
    device->deallocate(inner_forward_buffer_);
  }
  return status;
}

base::Status TensorRtInference::reshape(base::ShapeMap &shape_map) {
  bool flag = false;
  for (auto iter : shape_map) {
    auto tmp = current_shape_.find(iter.first);
    if (tmp != current_shape_.end()) {
      auto &shape = current_shape_[iter.first];
      if (base::shapeEqual(iter.second, shape)) {
        continue;
      } else {
        // current_shape_的修改
        current_shape_[iter.first] = iter.second;
        device::TensorDesc desc = current_input_tensors_[iter.first]->getDesc();
        desc.shape_ = iter.second;
        current_input_tensors_[iter.first]->justModify(desc);
        int idx = io_name_index_[iter.first];
        nvinfer1::Dims dims = TensorRtConvert::convertFromShape(iter.second);
        context_->setBindingDimensions(idx, dims);
      }
    } else {
      return base::kStatusCodeErrorInferenceTensorRt;
    }
  }

  return base::kStatusCodeOk;
}

int64_t TensorRtInference::getMemorySize() { return forward_memory_size_; }

base::Status TensorRtInference::setMemory(device::Buffer *buffer) {
  if (buffer && buffer->getSize() >= forward_memory_size_) {
    void *forward_memory_ = buffer->getPtr();
    context->setDeviceMemory(forward_memory_);
    return base::kStatusCodeOk;
  } else {
    return base::kStatusCodeErrorInferenceTensorRt;
  }
}

base::Status TensorRtInference::setInputTensor(
    const std::string &name,
    const std::shared_ptr<device::Tensor> input_tensor) {
  base::Status status = base::kStatusCodeOk;

  char *char_name = nullptr;
  if (!name.empty()) {
    char_name = const_cast<char *>(name.c_str());
  } else if (!input_tensor->getName().empty()) {
    char_name = const_cast<char *>(input_tensor->getName().c_str());
  } else {
    char_name =
        const_cast<char *>(current_input_tensors_.begin()->first.c_str());
  }

  device::Device *device = device::getDevice(inference_param_->device_type_);
  device::Buffer extern_input_buffer = input_tensor->getBuffer();
  base::DeviceType = extern_input_buffer->getDeviceType();
  device::Buffer internal_input_buffer =
      current_input_tensors_[char_name]->getBuffer();
  if (device::isHostDeviceType(device_type)) {
    device->upload(extern_input_buffer, internal_input_buffer);
  } else if (device_type == base::kDeviceTypeCodeCuda) {
    device->copy(extern_input_buffer, internal_input_buffer);
  } else {
    status = base::kStatusCodeErrorInferenceTensorRt;
  }

  return status;
}
/**
 * @brief
 *
 * @param name
 * @param config
 * [0] - device type
 * @return std::shared_ptr<device::Tensor>
 */
std::shared_ptr<device::Tensor> TensorRtInference::getOutputTensor(
    const std::string &name, std::vector<int32_t> config) {
  std::shared_ptr<device::Tensor> tensor;
  char *char_name = nullptr;
  if (!name.empty()) {
    char_name = const_cast<char *>(name.c_str());
  } else {
    char_name =
        const_cast<char *>(current_output_tensors_.begin()->first.c_str());
  }
  if (config.size() == 0) {
    tensors = current_output_tensors_[char_name];
  } else {
    device::Device *device = device::getDevice(inference_param_->device_type_);
    device::Buffer internal_output_buffer =
        current_output_tensors_[char_name]->getBuffer();
    if (config[0] == base::kDeviceTypeCodeCuda) {
      tensor.create(device, current_output_tensors_[char_name]->getDesc(),
                    current_output_tensors_[char_name]->getName());
      device::Buffer external_output_buffer = tensors->getBuffer();
      device->copy(internal_output_buffer, external_output_buffer);
    } else if (device::isHostDeviceType(config[1])) {
      base::DeviceType device_type = device::getDefaultHostDeviceType();
      device::Device *host_device = device::getDevice(device_type);
      tensor.create(host_device, current_output_tensors_[char_name]->getDesc(),
                    current_output_tensors_[char_name]->getName());
      device::Buffer external_output_buffer = tensors->getBuffer();
      device->download(internal_output_buffer, external_output_buffer);
    } else {
      return tensors;
    }
  }
  return tensors;
}

base::Status TensorRtInference::run() {
  base::Status status = base::kStatusCodeOk;
  device::Device *device = device::getDevice(inference_param_->device_type_);
  cudaStream_t stream_ = (cudaStream_t)device->getStream();
  if (!context_->enqueueV2(bindings_.data(), stream_, nullptr)) {
    return base::kStatusCodeErrorInferenceTensorRt;
  }
  status = device->synchronize();
  NNDEPLOY_RETURN_ON_NEQ(status, base::kStatusCodeOk);
  return status;
}

base::Status TensorRtInference::preRunWithOnnxModel(
    std::string model_buffer,
    TensorRtInferenceParam *tensor_rt_inference_param) {
  const auto explicit_batch =
      1U << static_cast<uint32_t>(
          nvinfer1::NetworkDefinitionCreationFlag::kEXPLICIT_BATCH);

  builder_ =
      UniquePtr<nvinfer1::IBuilder>(nvinfer1::createInferBuilder(logger_));
  if (!builder_) {
    // TODO: log
    return base::kStatusCodeErrorInferenceTensorRt;
  }
  network_ = UniquePtr<nvinfer1::INetworkDefinition>(
      builder_->createNetworkV2(explicit_batch));
  if (!network_) {
    // TODO: log
    return base::kStatusCodeErrorInferenceTensorRt;
  }
  parser_ = UniquePtr<nvonnxparser::IParser>(
      nvonnxparser::createParser(*network_, logger_));
  if (!parser_) {
    // TODO: log
    return base::kStatusCodeErrorInferenceTensorRt;
  }
  bool parser_flag = false;
  parser_flag = !parser_->parse(model_buffer.data(), model_buffer.size());
  if (parser_flag) {
    // TODO: log
    return base::kStatusCodeErrorInferenceTensorRt;
  }

  auto build_config = base::UniquePtr<nvinfer1::IBuilderInferenceParam>(
      builder_->createBuilderInferenceParam());
  if (!build_config) {
    // TODO: log
    return base::kStatusCodeErrorInferenceTensorRt;
  }

  if (tensor_rt_inference_param->precision_type_ == base::kPrecisionTypeFp16) {
    if (builder_->platformHasFastFp16()) {
      build_config->setFlag(nvinfer1::BuilderFlag::kFP16);
    }
  }

  if (context_) {
    context_.reset();
    engine_.reset();
  }

  builder_->setMaxBatchSize(tensor_rt_inference_param->max_batch_size_);
  build_config->setMaxWorkspaceSize(option_.workspace_size_);

  if (tensor_rt_inference_param->is_dynamic_shape_ && checkDynamicShape()) {
    auto profile = builder_->createOptimizationProfile();
    /**
     * @brief 出错机制
     * 如果min_shape_、opt_shape_、max_shape_中的shape不一致，会出错
     */
    for (const auto &item : min_shape_) {
      nvinfer1::Dims trt_shape = TensorRtConvert::convertFromShape(item.second);
      profile->setDimensions(item.first.c_str(),
                             nvinfer1::OptProfileSelector::kMIN, trt_shape)
    }
    if (opt_shape_.empty()) {
      for (const auto &item : max_shape_) {
        nvinfer1::Dims trt_shape =
            TensorRtConvert::convertFromShape(item.second);
        profile->setDimensions(item.first.c_str(),
                               nvinfer1::OptProfileSelector::kOPT, trt_shape)
      }
    } else {
      for (const auto &item : opt_shape_) {
        nvinfer1::Dims trt_shape =
            TensorRtConvert::convertFromShape(item.second);
        profile->setDimensions(item.first.c_str(),
                               nvinfer1::OptProfileSelector::kOPT, trt_shape)
      }
    }
    for (const auto &item : max_shape_) {
      nvinfer1::Dims trt_shape = TensorRtConvert::convertFromShape(item.second);
      profile->setDimensions(item.first.c_str(),
                             nvinfer1::OptProfileSelector::kMAX, trt_shape)
    }

    build_config->addOptimizationProfile(profile);
  }

  if (tensor_rt_inference_param->is_quant_ &&
      !tensor_rt_inference_param->int8_calibration_table_path_.empty()) {
    if (builder_->platformHasFastInt8()) {
      build_config->setFlag(nvinfer1::BuilderFlag::kINT8);
      build_config->setInt8Calibrator(
          tensor_rt_inference_param->int8_calibration_table_path_);
    }
  }

  base::UniquePtr<nvinfer1::IHostMemory> plan{
      builder_->buildSerializedNetwork(*network_, *build_config)};
  if (!plan) {
    // TODO: log
    return base::kStatusCodeErrorInferenceTensorRt;
  }
  if (!tensor_rt_inference_param->model_save_path_.empty()) {
    std::ofstream model_file(
        tensor_rt_inference_param->model_save_path_.c_str(),
        std::ios::binary | std::ios::out);
    if (!model_file) {
      return base::kStatusCodeErrorInferenceTensorRt;
    }
    model_file.write(static_cast<char *>(plan->data()), plan->size());
    model_file.close();
    return base::kStatusCodeOk;
  }

  base::UniquePtr<nvinfer1::IRuntime> runtime{
      nvinfer1::createInferRuntime(logger_)};
  if (!runtime) {
    // TODO: log
    return base::kStatusCodeErrorInferenceTensorRt;
  }

  engine_ = std::shared_ptr<nvinfer1::ICudaEngine>(
      runtime->deserializeCudaEngine(plan->data(), plan->size()));
  if (!engine_) {
    // TODO: log
    return base::kStatusCodeErrorInferenceTensorRt;
  }

  base::Status status = CreateExecuteContext();
  NNDEPLOY_RETURN_ON_NEQ(status, base::kStatusCodeOk);

  return base::kStatusCodeOk;
}

base::Status TensorRtInference::preRunWithTensorRtModel(
    std::string model_buffer,
    TensorRtInferenceParam *tensor_rt_inference_param) {
  UniquePtr<nvinfer1::IRuntime> runtime{nvinfer1::createInferRuntime(logger_)};
  if (!runtime) {
    return base::kStatusCodeErrorInferenceTensorRt;
  }

  engine_ = std::shared_ptr<nvinfer1::ICudaEngine>(
      runtime->deserializeCudaEngine(model_buffer.data(), model_buffer.size()));
  if (!engine_) {
    return base::kStatusCodeErrorInferenceTensorRt;
  }

  base::Status status = CreateExecuteContext();
  NNDEPLOY_RETURN_ON_NEQ(status, base::kStatusCodeOk);

  return base::kStatusCodeOk;
}

bool TensorRtInference::checkDynamicShape() { return true; }

base::Status TensorRtInference::CreateExecuteContext() {
  context_ = engine_->createExecutionContextWithoutDeviceMemory();
  if (!context_) {
    return base::kStatusCodeErrorInferenceTensorRt;
  }
  forward_memory_size_ =
      (std::max)(engine_->getDeviceMemorySize(), size_t(1024));
  if (tensor_rt_inference_param->share_memory_mode_ ==
      base::kShareMemoryTypeShareFromExternal) {
    ;
  } else {
    device::Device *device = device::getDevice(inference_param_->device_type_);
    inner_forward_buffer_ = device->allocate(forward_memory_size_);
    context_->setDeviceMemory(buffer_->getPtr());
  }
  return base::kStatusCodeOk;
}

}  // namespace inference
}  // namespace nndeploy
