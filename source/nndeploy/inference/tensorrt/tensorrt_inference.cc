
#include "nndeploy/inference/tensorrt/tensorrt_inference.h"

#include "nndeploy/base/shape.h"
#include "nndeploy/inference/tensorrt/tensorrt_convert.h"
#include "nndeploy/inference/tensorrt/tensorrt_inference_param.h"

namespace nndeploy {
namespace inference {

TypeInferenceRegister<TypeInferenceCreator<TensorRtInference>>
    g_tensorrt_inference_register(base::kInferenceTypeTensorRt);

class TensorRtLogger : public nvinfer1::ILogger {
 public:
  void log(nvinfer1::ILogger::Severity severity,
           const char *msg) noexcept override {
    // suppress info-level messages
#ifndef DEBUG
    if (severity == Severity::kINFO || severity == Severity::kVERBOSE) return;
#endif
    const char *skips[] = {
        "INVALID_ARGUMENT: Cannot find binding of given name",
        "Unused Input:",
        "Detected invalid timing cache",
        "unused or used only at compile-time",
    };

    std::string msg_str = std::string(msg);
    for (auto skip : skips) {
      if (msg_str.find(skip) != std::string::npos) {
        return;
      }
    }
    switch (severity) {
      case Severity::kINTERNAL_ERROR:
        std::cerr << "INTERNAL_ERROR: ";
        break;
      case Severity::kERROR:
        std::cerr << "ERROR: ";
        break;
      case Severity::kWARNING:
        std::cerr << "WARNING: ";
        break;
      case Severity::kINFO:
        break;
      case Severity::kVERBOSE:
        std::cerr << "VERBOSE: ";
        break;
      default:
        break;
    }
    std::cerr << msg << std::endl;
  }
};

static TensorRtLogger g_logger;

TensorRtInference::TensorRtInference(base::InferenceType type)
    : Inference(type) {}

TensorRtInference::~TensorRtInference() {}

base::Status TensorRtInference::init() {
  base::Status status = base::kStatusCodeOk;

  is_share_command_queue_ = true;

  std::string model_buffer;
  TensorRtInferenceParam *tensorrt_inference_param =
      dynamic_cast<TensorRtInferenceParam *>(inference_param_);
  if (tensorrt_inference_param->is_path_) {
    model_buffer = base::openFile(tensorrt_inference_param->model_value_[0]);
  } else {
    model_buffer = tensorrt_inference_param->model_value_[0];
  }

  if (tensorrt_inference_param->model_type_ == base::kModelTypeOnnx) {
    status = initWithOnnxModel(model_buffer, tensorrt_inference_param);
    NNDEPLOY_RETURN_ON_NEQ(status, base::kStatusCodeOk,
                           "initWithOnnxModel failed");
  } else if (tensorrt_inference_param->model_type_ ==
             base::kModelTypeTensorRt) {
    status = initWithTensorRtModel(model_buffer, tensorrt_inference_param);
    NNDEPLOY_RETURN_ON_NEQ(status, base::kStatusCodeOk,
                           "initWithTensorRtModel failed");
  } else {
    NNDEPLOY_LOGE("not support this model type(%d)!\n",
                  tensorrt_inference_param->model_type_);
    return base::kStatusCodeErrorInferenceTensorRt;
  }

  for (auto i = 0; i < getNbBindings(); ++i) {
    std::string name = std::string(getBindingName(i));
    io_name_index_[name] = i;
    io_index_name_[i] = name;
  }

  base::ShapeMap current_shape;
  for (auto i = 0; i < getNbBindings(); ++i) {
    if (bindingIsInput(i)) {
      std::string name = std::string(getBindingName(i));
      auto shape = TensorRtConvert::convertToShape(getBindingDimensions(i));
      current_shape.insert({name, shape});
    }
  }
  for (auto iter : tensorrt_inference_param->max_shape_) {
    auto tmp = current_shape.find(iter.first);
    if (tmp != current_shape.end()) {
      auto &shape = current_shape[iter.first];
      if (base::shapeEqual(iter.second, shape)) {
        continue;
      } else {
        int idx = io_name_index_[iter.first];
        nvinfer1::Dims dims = TensorRtConvert::convertFromShape(iter.second);
        setBindingDimensions(idx, dims);
      }
    } else {
      NNDEPLOY_LOGE("reshape failed, not found input tensor(%s)!\n",
                    iter.first.c_str());
      return base::kStatusCodeErrorInferenceTensorRt;
    }
  }

  device::Device *device = device::getDevice(inference_param_->device_type_);
  auto num_binds = getNbBindings();
  bindings_.resize(num_binds);
  for (auto i = 0; i < num_binds; ++i) {
    std::string name = std::string(getBindingName(i));
    base::IntVector shape =
        TensorRtConvert::convertToShape(getBindingDimensions(i));
    base::DataType data_type =
        TensorRtConvert::convertToDataType(getBindingDataType(i));
    base::DataFormat data_format =
        TensorRtConvert::convertToDataFormat(getBindingFormat(i));

    if (bindingIsInput(i)) {
      device::TensorDesc desc;
      desc.data_type_ = data_type;
      desc.data_format_ = data_format;
      desc.shape_ = shape;
      device::Tensor *max_input_tensor = new device::Tensor(device, desc, name);
      max_input_tensors_.insert({name, max_input_tensor});

      device::Buffer *max_input_buffer = max_input_tensor->getBuffer();
      device::Tensor *current_input_tensor =
          new device::Tensor(desc, max_input_buffer, name);
      input_tensors_.insert({name, current_input_tensor});

      // bindings_[i] = max_input_buffer->getPtr();
    } else {
      device::TensorDesc desc;
      desc.data_type_ = data_type;
      desc.data_format_ = data_format;
      desc.shape_ = shape;
      device::Tensor *max_output_tensor =
          new device::Tensor(device, desc, name);
      max_output_tensors_.insert({name, max_output_tensor});

      device::Buffer *max_output_buffer = max_output_tensor->getBuffer();
      device::Tensor *current_output_tensor =
          new device::Tensor(desc, max_output_buffer, name);
      output_tensors_.insert({name, current_output_tensor});

      // bindings_[i] = max_output_buffer->getPtr();
    }
  }

  return status;
}

base::Status TensorRtInference::deinit() {
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
  device::Device *device = device::getDevice(inference_param_->device_type_);
  if (inner_forward_buffer_ != nullptr) {
    device->deallocate(inner_forward_buffer_);
  }
  return status;
}

base::Status TensorRtInference::reshape(base::ShapeMap &shape_map) {
  base::ShapeMap current_shape;
  auto num_binds = getNbBindings();
  for (auto i = 0; i < num_binds; ++i) {
    if (bindingIsInput(i)) {
      std::string name = std::string(getBindingName(i));
      auto shape = TensorRtConvert::convertToShape(getBindingDimensions(i));
      current_shape.insert({name, shape});
    }
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
        int idx = io_name_index_[iter.first];
        nvinfer1::Dims dims = TensorRtConvert::convertFromShape(iter.second);
        setBindingDimensions(idx, dims);
      }
    } else {
      NNDEPLOY_LOGE("reshape failed, not found input tensor(%s)!\n",
                    iter.first.c_str());
      return base::kStatusCodeErrorInferenceTensorRt;
    }
  }

  return base::kStatusCodeOk;
}

int64_t TensorRtInference::getMemorySize() { return forward_memory_size_; }

base::Status TensorRtInference::setMemory(device::Buffer *buffer) {
  if (buffer && buffer->getSize() >= forward_memory_size_) {
    void *forward_memory_ = buffer->getPtr();
    context_->setDeviceMemory(forward_memory_);
    return base::kStatusCodeOk;
  } else {
    NNDEPLOY_LOGE(
        "setMemory failed, buffer is nullptr or buffer size is less "
        "than forward_memory_size_!\n");
    return base::kStatusCodeErrorInferenceTensorRt;
  }
}

base::Status TensorRtInference::run() {
  base::Status status = base::kStatusCodeOk;
  device::Device *device = device::getDevice(inference_param_->device_type_);
  // inputs
  for (auto iter : external_input_tensors_) {
    device::Tensor *external_tensor = iter.second;
    device::Buffer *extern_buffer = external_tensor->getBuffer();
    device::Tensor *internal_tensor = input_tensors_[iter.first];
    device::Buffer *internal_buffer = internal_tensor->getBuffer();
    base::DeviceType device_type = extern_buffer->getDeviceType();
    if (device::isHostDeviceType(device_type)) {
      device->upload(extern_buffer, internal_buffer);
    } else if (device_type == device->getDeviceType()) {
      device->copy(extern_buffer, internal_buffer);
    } else {
      NNDEPLOY_LOGE("run failed, device type is not supported!\n");
      return base::kStatusCodeErrorInferenceTensorRt;
    }
  }
  // forward
  cudaStream_t stream_ = (cudaStream_t)device->getCommandQueue();
#ifdef TENSORRT_MAJOR_8_MINOR_5
  for (auto iter : max_input_tensors_) {
    void *data = iter.second->getBuffer()->getPtr();
    if (!context_->setTensorAddress(iter.first.c_str(), data)) {
      NNDEPLOY_LOGE("Fail to setTensorAddress [%s]!\n", iter.first.c_str());
      return base::kStatusCodeErrorInferenceTensorRt;
    }
  }
  for (auto iter : max_output_tensors_) {
    void *data = iter.second->getBuffer()->getPtr();
    if (!context_->setTensorAddress(iter.first.c_str(), data)) {
      NNDEPLOY_LOGE("Fail to setTensorAddress [%s]!\n", iter.first.c_str());
      return base::kStatusCodeErrorInferenceTensorRt;
    }
  }
  if (!context_->enqueueV3(stream_)) {
    NNDEPLOY_LOGE("Fail to enqueueV3!\n");
    return base::kStatusCodeErrorInferenceTensorRt;
  }
#else
  auto num_binds = getNbBindings();
  if (bindingIsInput(i)) {
    auto max_input_buffer = max_input_tensors_[io_index_name_[i]]->getBuffer();
    bindings_[i] = max_input_buffer->getPtr();
  } else {
    auto max_output_buffer =
        max_output_tensors_[io_index_name_[i]]->getBuffer();
    bindings_[i] = max_output_buffer->getPtr();
  }
  if (!context_->enqueueV2(bindings_.data(), stream_, nullptr)) {
    NNDEPLOY_LOGE("Fail to enqueueV2!\n");
    return base::kStatusCodeErrorInferenceTensorRt;
  }
#endif
  status = device->synchronize();
  NNDEPLOY_RETURN_ON_NEQ(status, base::kStatusCodeOk, "synchronize failed");
  // outputs
  for (auto iter : external_output_tensors_) {
    device::Tensor *external_tensor = iter.second;
    device::Buffer *extern_buffer = external_tensor->getBuffer();
    device::Tensor *internal_tensor = output_tensors_[iter.first];
    device::Buffer *internal_buffer = internal_tensor->getBuffer();
    base::DeviceType device_type = extern_buffer->getDeviceType();
    if (device::isHostDeviceType(device_type)) {
      device->download(internal_buffer, extern_buffer);
    } else if (device_type == device->getDeviceType()) {
      device->copy(internal_buffer, extern_buffer);
    } else {
      NNDEPLOY_LOGE("run failed, device type is not supported!\n");
      return base::kStatusCodeErrorInferenceTensorRt;
    }
  }
  return status;
}

base::Status TensorRtInference::initWithOnnxModel(
    const std::string &model_buffer,
    TensorRtInferenceParam *tensorrt_inference_param) {
  const auto explicit_batch =
      1U << static_cast<uint32_t>(
          nvinfer1::NetworkDefinitionCreationFlag::kEXPLICIT_BATCH);

  builder_ = base::UniquePtr<nvinfer1::IBuilder>(
      nvinfer1::createInferBuilder(g_logger));
  if (!builder_) {
    NNDEPLOY_LOGE("createInferBuilder failed!\n");
    return base::kStatusCodeErrorInferenceTensorRt;
  }
  network_ = base::UniquePtr<nvinfer1::INetworkDefinition>(
      builder_->createNetworkV2(explicit_batch));
  if (!network_) {
    NNDEPLOY_LOGE("createNetworkV2 failed!\n");
    return base::kStatusCodeErrorInferenceTensorRt;
  }
  parser_ = base::UniquePtr<nvonnxparser::IParser>(
      nvonnxparser::createParser(*network_, g_logger));
  if (!parser_) {
    NNDEPLOY_LOGE("createParser failed!\n");
    return base::kStatusCodeErrorInferenceTensorRt;
  }
  bool parser_flag = false;
  parser_flag = !parser_->parse(model_buffer.data(), model_buffer.size());
  if (parser_flag) {
    NNDEPLOY_LOGE("parse failed!\n");
    return base::kStatusCodeErrorInferenceTensorRt;
  }

  auto build_config = base::UniquePtr<nvinfer1::IBuilderConfig>(
      builder_->createBuilderConfig());
  if (!build_config) {
    NNDEPLOY_LOGE("createBuilderInferenceParam failed!\n");
    return base::kStatusCodeErrorInferenceTensorRt;
  }

  if (tensorrt_inference_param->precision_type_ == base::kPrecisionTypeFp16) {
    if (builder_->platformHasFastFp16()) {
      build_config->setFlag(nvinfer1::BuilderFlag::kFP16);
    }
  }

  if (context_) {
    context_.reset();
    engine_.reset();
  }
#if NV_TENSORRT_MAJOR < 8 && NV_TENSORRT_MINOR < 4
  builder_->setMaxBatchSize(tensorrt_inference_param->max_batch_size_);
#endif
#if NV_TENSORRT_MAJOR < 8 && NV_TENSORRT_MINOR < 4
  build_config->setMaxWorkspaceSize(tensorrt_inference_param->workspace_size_);
#else
  build_config->setMemoryPoolLimit(nvinfer1::MemoryPoolType::kWORKSPACE,
                                   tensorrt_inference_param->workspace_size_);
#endif

  if (tensorrt_inference_param->is_dynamic_shape_ && checkDynamicShape()) {
    auto profile = builder_->createOptimizationProfile();
    /**
     * @brief 出错机制
     * 如果min_shape_、opt_shape_、max_shape_中的shape不一致，会出错
     */
    for (const auto &item : tensorrt_inference_param->min_shape_) {
      nvinfer1::Dims trt_shape = TensorRtConvert::convertFromShape(item.second);
      profile->setDimensions(item.first.c_str(),
                             nvinfer1::OptProfileSelector::kMIN, trt_shape);
    }
    if (tensorrt_inference_param->opt_shape_.empty()) {
      for (const auto &item : tensorrt_inference_param->max_shape_) {
        nvinfer1::Dims trt_shape =
            TensorRtConvert::convertFromShape(item.second);
        profile->setDimensions(item.first.c_str(),
                               nvinfer1::OptProfileSelector::kOPT, trt_shape);
      }
    } else {
      for (const auto &item : tensorrt_inference_param->opt_shape_) {
        nvinfer1::Dims trt_shape =
            TensorRtConvert::convertFromShape(item.second);
        profile->setDimensions(item.first.c_str(),
                               nvinfer1::OptProfileSelector::kOPT, trt_shape);
      }
    }
    for (const auto &item : tensorrt_inference_param->max_shape_) {
      nvinfer1::Dims trt_shape = TensorRtConvert::convertFromShape(item.second);
      profile->setDimensions(item.first.c_str(),
                             nvinfer1::OptProfileSelector::kMAX, trt_shape);
    }

    build_config->addOptimizationProfile(profile);
  }

  if (tensorrt_inference_param->is_quant_ &&
      !tensorrt_inference_param->int8_calibration_table_path_.empty()) {
    if (builder_->platformHasFastInt8()) {
      build_config->setFlag(nvinfer1::BuilderFlag::kINT8);
      // build_config->setInt8Calibrator(
      //     tensorrt_inference_param->int8_calibration_table_path_);
    }
  }

  base::UniquePtr<nvinfer1::IHostMemory> plan{
      builder_->buildSerializedNetwork(*network_, *build_config)};
  if (!plan) {
    NNDEPLOY_LOGE("buildSerializedNetwork failed!\n");
    return base::kStatusCodeErrorInferenceTensorRt;
  }
  if (!tensorrt_inference_param->model_save_path_.empty()) {
    std::ofstream model_file(tensorrt_inference_param->model_save_path_.c_str(),
                             std::ios::binary | std::ios::out);
    if (!model_file) {
      return base::kStatusCodeErrorInferenceTensorRt;
    }
    model_file.write(static_cast<char *>(plan->data()), plan->size());
    model_file.close();
    return base::kStatusCodeOk;
  }

  runtime_ = base::UniquePtr<nvinfer1::IRuntime>(
      nvinfer1::createInferRuntime(g_logger));
  if (!runtime_) {
    NNDEPLOY_LOGE("createInferRuntime failed!\n");
    return base::kStatusCodeErrorInferenceTensorRt;
  }

  engine_ = std::shared_ptr<nvinfer1::ICudaEngine>(
      runtime_->deserializeCudaEngine(plan->data(), plan->size()));
  if (!engine_) {
    NNDEPLOY_LOGE("deserializeCudaEngine failed!\n");
    return base::kStatusCodeErrorInferenceTensorRt;
  }

  base::Status status = CreateExecuteContext();
  NNDEPLOY_RETURN_ON_NEQ(status, base::kStatusCodeOk,
                         "CreateExecuteContext failed");

  return base::kStatusCodeOk;
}

base::Status TensorRtInference::initWithTensorRtModel(
    const std::string &model_buffer,
    TensorRtInferenceParam *tensorrt_inference_param) {
  runtime_ = base::UniquePtr<nvinfer1::IRuntime>(
      nvinfer1::createInferRuntime(g_logger));
  if (!runtime_) {
    NNDEPLOY_LOGE("createInferRuntime failed!\n");
    return base::kStatusCodeErrorInferenceTensorRt;
  }

  engine_ =
      std::shared_ptr<nvinfer1::ICudaEngine>(runtime_->deserializeCudaEngine(
          model_buffer.data(), model_buffer.size()));
  if (!engine_) {
    NNDEPLOY_LOGE("deserializeCudaEngine failed!\n");
    return base::kStatusCodeErrorInferenceTensorRt;
  }

  base::Status status = CreateExecuteContext();
  NNDEPLOY_RETURN_ON_NEQ(status, base::kStatusCodeOk,
                         "CreateExecuteContext failed");

  return base::kStatusCodeOk;
}

bool TensorRtInference::checkDynamicShape() { return true; }

base::Status TensorRtInference::CreateExecuteContext() {
  context_ = std::shared_ptr<nvinfer1::IExecutionContext>(
      engine_->createExecutionContextWithoutDeviceMemory());
  if (!context_) {
    return base::kStatusCodeErrorInferenceTensorRt;
  }
  forward_memory_size_ =
      (std::max)(engine_->getDeviceMemorySize(), size_t(1024));
  if (inference_param_->share_memory_mode_ ==
      base::kShareMemoryTypeShareFromExternal) {
    ;
  } else {
    device::Device *device = device::getDevice(inference_param_->device_type_);
    inner_forward_buffer_ = device->allocate(forward_memory_size_);
    context_->setDeviceMemory(inner_forward_buffer_->getPtr());
  }
  return base::kStatusCodeOk;
}

char const *TensorRtInference::getBindingName(int32_t binding_index) {
#ifdef TENSORRT_MAJOR_8_MINOR_5
  return engine_->getIOTensorName(binding_index);
#else
  return engine_->getBindingName(binding_index);
#endif  // TENSORRT_MAJOR_8_MINOR_5
}

int TensorRtInference::getNbBindings() {
#ifdef TENSORRT_MAJOR_8_MINOR_5
  return engine_->getNbIOTensors();
#else
  return engine_->getNbBindings();
#endif  // TENSORRT_MAJOR_8_MINOR_5
}

bool TensorRtInference::bindingIsInput(int32_t binding_index) {
#ifdef TENSORRT_MAJOR_8_MINOR_5
  char const *name = engine_->getIOTensorName(binding_index);
  nvinfer1::TensorIOMode io_mode = engine_->getTensorIOMode(name);
  return io_mode == nvinfer1::TensorIOMode::kINPUT;
#else
  return engine_->bindingIsInput(binding_index);
#endif  // TENSORRT_MAJOR_8_MINOR_5
}

nvinfer1::DataType TensorRtInference::getBindingDataType(
    int32_t binding_index) {
#ifdef TENSORRT_MAJOR_8_MINOR_5
  char const *name = engine_->getIOTensorName(binding_index);
  return engine_->getTensorDataType(name);
#else
  return engine_->getBindingDataType(binding_index);
#endif  // TENSORRT_MAJOR_8_MINOR_5
}

nvinfer1::TensorFormat TensorRtInference::getBindingFormat(
    int32_t binding_index) {
#ifdef TENSORRT_MAJOR_8_MINOR_5
  char const *name = engine_->getIOTensorName(binding_index);
  return engine_->getTensorFormat(name);
#else
  return engine_->getBindingFormat(binding_index);
#endif  // TENSORRT_MAJOR_8_MINOR_5
}

nvinfer1::Dims TensorRtInference::getBindingDimensions(int32_t binding_index) {
#ifdef TENSORRT_MAJOR_8_MINOR_5
  char const *name = engine_->getIOTensorName(binding_index);
  return engine_->getTensorShape(name);
#else
  return engine_->getBindingDimensions(binding_index);
#endif  // TENSORRT_MAJOR_8_MINOR_5
}

bool TensorRtInference::setBindingDimensions(int32_t binding_index,
                                             nvinfer1::Dims dimensions) {
#ifdef TENSORRT_MAJOR_8_MINOR_5
  char const *name = engine_->getIOTensorName(binding_index);
  return context_->setInputShape(name, dimensions);
#else
  return context_->setBindingDimensions(binding_index, dimensions);
#endif  // TENSORRT_MAJOR_8_MINOR_5
}

}  // namespace inference
}  // namespace nndeploy
