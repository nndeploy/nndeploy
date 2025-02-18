#include "nndeploy/inference/snpe/snpe_inference.h"

namespace nndeploy {
namespace inference {

TypeInferenceRegister<TypeInferenceCreator<SnpeInference>>
    g_snpe_inference_register(base::kInferenceTypeSnpe);

SnpeInference::SnpeInference(base::InferenceType type) : Inference(type) {}
SnpeInference::~SnpeInference() {}

base::Status SnpeInference::init() {
  base::Status status = base::kStatusCodeOk;

  // if (device::isHostDeviceType(inference_param_->device_type_)) {
  //   is_share_context_ = true;
  // } else {
  //   is_share_context_ = false;
  // }

  SnpeInferenceParam* snpe_inference_param =
      dynamic_cast<SnpeInferenceParam*>(inference_param_);

  zdl::DlSystem::Runtime_t runtime;
  std::string snpe_runtime = snpe_inference_param->runtime_;
  std::cout << "snpe_runtime is : [" << snpe_runtime << "]." << std::endl;
  if (snpe_runtime == "cpu") {
    runtime = zdl::DlSystem::Runtime_t::CPU;
  } else if (snpe_runtime == "gpu") {
    runtime = zdl::DlSystem::Runtime_t::GPU;
  } else if (snpe_runtime == "dsp") {
    runtime = zdl::DlSystem::Runtime_t::DSP;
  } else if (snpe_runtime == "aip") {
    runtime = zdl::DlSystem::Runtime_t::AIP_FIXED8_TF;
  } else {
    std::cout << "SNPE runtime option[" << snpe_runtime
              << "] is not valid! Defaulting to the CPU runtime." << std::endl;
    runtime = zdl::DlSystem::Runtime_t::CPU;
  }
  if (!zdl::SNPE::SNPEFactory::isRuntimeAvailable(runtime)) {
    std::cout << "Selected runtime[" << snpe_runtime
              << "] not available! Please check your environment and runtime "
                 "setting in .json!"
              << std::endl;
    std::abort();
  }

  zdl::DlSystem::PerformanceProfile_t perf_mode;
  int32_t snpe_perf_mode = snpe_inference_param->perf_mode_;
  std::cout << "snpe_perf_mode is : [" << snpe_perf_mode << "]." << std::endl;
  switch (snpe_perf_mode) {
    case 0:
      perf_mode = zdl::DlSystem::PerformanceProfile_t::BALANCED;
      break;
    case 1:
      perf_mode = zdl::DlSystem::PerformanceProfile_t::HIGH_PERFORMANCE;
      break;
    case 2:
      perf_mode = zdl::DlSystem::PerformanceProfile_t::POWER_SAVER;
      break;
    case 3:
      perf_mode = zdl::DlSystem::PerformanceProfile_t::SYSTEM_SETTINGS;
      break;
    case 4:
      perf_mode =
          zdl::DlSystem::PerformanceProfile_t::SUSTAINED_HIGH_PERFORMANCE;
      break;
    case 5:
      perf_mode = zdl::DlSystem::PerformanceProfile_t::BURST;
      break;
    default:
      std::cout << "SNPE performance mode[" << snpe_perf_mode
                << "] is not valid!" << std::endl;
      std::abort();
  }

  zdl::DlSystem::ProfilingLevel_t profiling_level;
  int32_t snpe_profiling_level = snpe_inference_param->profiling_level_;
  std::cout << "snpe_profiling_level is : [" << snpe_profiling_level << "]."
            << std::endl;
  switch (snpe_profiling_level) {
    case 0:
      profiling_level = zdl::DlSystem::ProfilingLevel_t::OFF;
      break;
    case 1:
      profiling_level = zdl::DlSystem::ProfilingLevel_t::BASIC;
      break;
    case 2:
      profiling_level = zdl::DlSystem::ProfilingLevel_t::DETAILED;
      break;
    default:
      std::cout << "SNPE profiling level[" << snpe_profiling_level
                << "] is not valid!" << std::endl;
      std::abort();
  }

  int32_t buffer_type = snpe_inference_param->buffer_type_;
  std::cout << "buffer_type is : [" << buffer_type << "]." << std::endl;
  int32_t bit_width = 0;
  switch (buffer_type) {
    case 0:
      buffer_type_ = USERBUFFER_FLOAT;
      break;
    case 1:
      buffer_type_ = USERBUFFER_TF8;
      break;
    case 2:
      buffer_type_ = ITENSOR;
      break;
    case 3:
      buffer_type_ = USERBUFFER_TF16;
      break;
    default:
      std::cout << "The buffer type is not supported." << std::endl;
      break;
  }
  bool use_user_supplied_buffers =
      (buffer_type_ == USERBUFFER_FLOAT || buffer_type_ == USERBUFFER_TF8 ||
       buffer_type_ == USERBUFFER_TF16 || buffer_type_ == ITENSOR);

  std::string modelfile_path = snpe_inference_param->model_value_[0];
  std::ifstream dlcFile(modelfile_path);
  if (!dlcFile) {
    std::cout
        << "DLC file[" << modelfile_path
        << "] not valid. Please ensure that you have provided a valid dlc file!"
        << std::endl;
    std::abort();
  }

  std::unique_ptr<zdl::DlContainer::IDlContainer> container =
      zdl::DlContainer::IDlContainer::open(
          zdl::DlSystem::String(modelfile_path.c_str()));
  if (container == nullptr) {
    std::cout << "Error while opening the container file[" << modelfile_path
              << "]." << std::endl;
    return base::kStatusCodeErrorInferenceSnpe;
  }

  zdl::DlSystem::UDLFactoryFunc udl_func = MyUDLFactory;
  zdl::DlSystem::UDLBundle udl_bundle;
  udl_bundle.cookie = (void*)0xdeadbeaf,
  udl_bundle.func = udl_func;  // 0xdeadbeaf to test cookie
  zdl::DlSystem::PlatformConfig platform_config;
  bool usingInitCaching = false;

  zdl::SNPE::SNPEBuilder snpe_builder(container.get());

  zdl::DlSystem::StringList outputLayerNames = {};
  std::vector<std::string> output_layer_names =
      snpe_inference_param->output_layer_names_;
  for (int i = 0; i < output_layer_names.size(); i++) {
    std::cout << "output_layer_names[" << i
              << "] is : " << output_layer_names[i] << std::endl;
    outputLayerNames.append(output_layer_names[i].c_str());
  }

  if (outputLayerNames.size() != 0) {
    snpe_ = snpe_builder.setOutputLayers(outputLayerNames)
                .setRuntimeProcessor(runtime)
                .setUdlBundle(udl_bundle)
                .setPerformanceProfile(perf_mode)
                .setProfilingLevel(profiling_level)
                .setUseUserSuppliedBuffers(use_user_supplied_buffers)
                .setPlatformConfig(platform_config)
                .setInitCacheMode(usingInitCaching)
                .build();
  }

  if (snpe_ == nullptr) {
    std::cout << "Error while building SNPE object." << std::endl;
    return base::kStatusCodeErrorInferenceSnpe;
  }

  if (buffer_type == USERBUFFER_FLOAT) {
    createInputBufferMap(input_map_, application_input_buffers_,
                         snpe_user_input_buffers_, snpe_, false, bit_width);
    createOutputBufferMap(output_map_, application_output_buffers_,
                          snpe_user_output_buffers_, snpe_, false, bit_width);
  } else if (buffer_type == USERBUFFER_TF8 || buffer_type == USERBUFFER_TF16) {
    createInputBufferMap(input_map_, application_input_buffers_,
                         snpe_user_input_buffers_, snpe_, true, bit_width);
    createOutputBufferMap(output_map_, application_output_buffers_,
                          snpe_user_output_buffers_, snpe_, true, bit_width);
  } else if (buffer_type == ITENSOR) {
    std::cout << "The ITensor data type is not supported!" << std::endl;
  }

  status = allocateInputOutputTensor();

  return status;
}

base::Status SnpeInference::deinit() {
  base::Status status = deallocateInputOutputTensor();
  NNDEPLOY_RETURN_ON_NEQ(status, base::kStatusCodeOk,
                         "deallocateInputOutputTensor failed!!\n");

  return base::kStatusCodeOk;
}

base::Status SnpeInference::run() {
  for (auto iter : external_input_tensors_) {
    device::Tensor* tensor = iter.second;
    int n = tensor->getBatch();
    int h = tensor->getHeight();
    int w = tensor->getWidth();
    int c = tensor->getChannel();
    int elemsize = n * h * w * c * sizeof(float);
    std::string name = iter.first;
    float* data = static_cast<float*>(tensor->getData());
    memcpy(application_input_buffers_.at(name).data(), data, elemsize);

    bool execStatus = snpe_->execute(input_map_, output_map_);
    if (execStatus != true) {
      std::cout << "Error while executing the network! Error info: "
                << zdl::DlSystem::getLastErrorString() << std::endl;

      return base::kStatusCodeErrorInferenceSnpe;
    }
  }

  return base::kStatusCodeOk;
}

base::Status SnpeInference::reshape(base::ShapeMap& shape_map) {
  return base::kStatusCodeOk;
}

device::Tensor* SnpeInference::getOutputTensorAfterRun(
    const std::string& name, base::DeviceType device_type, bool is_copy,
    base::DataFormat data_format) {
  device::Device* device = device::getDevice(device_type);
  device::Tensor* internal_tensor = output_tensors_[name];
  device::TensorDesc desc = internal_tensor->getDesc();
  bool flag = is_copy || (internal_tensor->getDevice() != device);
  if (flag) {
    device::Tensor* output_tensor = new device::Tensor(device, desc, name);
    internal_tensor->getBuffer()->copyTo(output_tensor->getBuffer());
    return output_tensor;
  } else {
    device::Tensor* output_tensor =
        new device::Tensor(desc, internal_tensor->getBuffer(), name);
    return output_tensor;
  }
}

base::Status SnpeInference::allocateInputOutputTensor() {
  /* debug info: input tensor name, size; output tensor name, size */
  const zdl::DlSystem::Optional<zdl::DlSystem::StringList>&
      input_tensor_names_opt = snpe_->getInputTensorNames();
  const zdl::DlSystem::Optional<zdl::DlSystem::StringList>&
      output_tensor_names_opt = snpe_->getOutputTensorNames();
  const zdl::DlSystem::Optional<zdl::DlSystem::StringList>&
      output_layer_names_opt = snpe_->getOutputLayerNames();
  const zdl::DlSystem::StringList& input_tensor_names = *input_tensor_names_opt;
  const zdl::DlSystem::StringList& output_tensor_names =
      *output_tensor_names_opt;
  const zdl::DlSystem::StringList& output_layer_names = *output_layer_names_opt;

  device::Device* host_device = device::getDefaultHostDevice();
  device::Device* device = nullptr;
  if (device::isHostDeviceType(inference_param_->device_type_)) {
    device = device::getDevice(inference_param_->device_type_);
  }

  for (const char* name : input_tensor_names) {
    const zdl::DlSystem::Optional<zdl::DlSystem::TensorShape>& tensorShapeOpt =
        snpe_->getInputDimensions(name);
    const zdl::DlSystem::TensorShape& tensorShape = *tensorShapeOpt;

    const zdl::DlSystem::Dimension* dims = tensorShape.getDimensions();
    size_t rank = tensorShape.rank();
    base::IntVector shape = SnpeConvert::convertToShape(dims, rank);
    base::DataType data_type = SnpeConvert::convertToDataType(buffer_type_);
    base::DataFormat data_format = SnpeConvert::convertToDataFormat();

    device::TensorDesc desc;
    desc.data_type_ = data_type;
    desc.data_format_ = data_format;
    desc.shape_ = shape;
    desc.stride_ = base::SizeVector();

    device::Tensor* input_tensor = nullptr;
    void* data_ptr =
        reinterpret_cast<void*>(&application_input_buffers_.at(name)[0]);
    base::IntVector memory_config = base::IntVector();
    input_tensor =
        new device::Tensor(device, desc, data_ptr, name, memory_config);

    input_tensors_.insert({name, input_tensor});
  }
  for (const char* name : output_tensor_names) {
    auto bufferAttributesOpt = snpe_->getInputOutputBufferAttributes(name);
    const zdl::DlSystem::TensorShape& tensorShape =
        (*bufferAttributesOpt)->getDims();

    const zdl::DlSystem::Dimension* dims = tensorShape.getDimensions();
    size_t rank = tensorShape.rank();
    base::IntVector shape = SnpeConvert::convertToShape(dims, rank);
    base::DataType data_type = SnpeConvert::convertToDataType(buffer_type_);
    base::DataFormat data_format = SnpeConvert::convertToDataFormat();

    device::TensorDesc desc;
    desc.data_type_ = data_type;
    desc.data_format_ = data_format;
    desc.shape_ = shape;
    desc.stride_ = base::SizeVector();

    device::Tensor* output_tensor = nullptr;
    void* data_ptr =
        reinterpret_cast<void*>(&application_output_buffers_.at(name)[0]);
    base::IntVector memory_config = base::IntVector();
    output_tensor =
        new device::Tensor(device, desc, data_ptr, name, memory_config);

    output_tensors_.insert({name, output_tensor});
  }

  return base::kStatusCodeOk;
}

base::Status SnpeInference::deallocateInputOutputTensor() {
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

size_t SnpeInference::calcSizeFromDims(const zdl::DlSystem::Dimension* dims,
                                       size_t rank, size_t elementSize) {
  if (rank == 0) return 0;
  size_t size = elementSize;

  while (rank--) {
    (*dims == 0) ? size *= resizable_dim : size *= *dims;
    dims++;
  }

  return size;
}

void SnpeInference::setResizableDim(size_t resizableDim) {
  resizable_dim = resizableDim;
}

size_t SnpeInference::getResizableDim() { return resizable_dim; }

void SnpeInference::createUserBuffer(
    zdl::DlSystem::UserBufferMap& userBufferMap,
    std::unordered_map<std::string, std::vector<uint8_t>>& applicationBuffers,
    std::vector<std::unique_ptr<zdl::DlSystem::IUserBuffer>>&
        snpeUserBackedBuffers,
    std::unique_ptr<zdl::SNPE::SNPE>& snpe, const char* name,
    const bool isTfNBuffer, int bitWidth) {
  // get attributes of buffer by name
  auto bufferAttributesOpt = snpe->getInputOutputBufferAttributes(name);
  if (!bufferAttributesOpt) {
    throw std::runtime_error(
        std::string("Error obtaining attributes for input tensor ") + name);
  }

  // calculate the size of buffer required by the input tensor
  const zdl::DlSystem::TensorShape& bufferShape =
      (*bufferAttributesOpt)->getDims();

  size_t bufferElementSize = 0;
  if (isTfNBuffer) {
    bufferElementSize = sizeof(uint8_t);
  } else {
    bufferElementSize = sizeof(float);
  }

  // Calculate the stride based on buffer strides.
  // Note: Strides = Number of bytes to advance to the next element in each
  // dimension. For example, if a float tensor of dimension 2x4x3 is tightly
  // packed in a buffer of 96 bytes, then the strides would be (48,12,4) Note:
  // Buffer stride is usually known and does not need to be calculated.
  std::vector<size_t> strides(bufferShape.rank());
  strides[strides.size() - 1] = bufferElementSize;
  size_t stride = strides[strides.size() - 1];
  for (size_t i = bufferShape.rank() - 1; i > 0; i--) {
    (bufferShape[i] == 0) ? stride *= getResizableDim()
                          : stride *= bufferShape[i];
    strides[i - 1] = stride;
  }

  size_t bufSize = calcSizeFromDims(bufferShape.getDimensions(),
                                    bufferShape.rank(), bufferElementSize);

  // set the buffer encoding type
  std::unique_ptr<zdl::DlSystem::UserBufferEncoding> userBufferEncoding;
  if (isTfNBuffer) {
    userBufferEncoding =
        std::move(std::unique_ptr<zdl::DlSystem::UserBufferEncodingTf8>(
            new zdl::DlSystem::UserBufferEncodingTf8(0, 1.0)));
  } else {
    userBufferEncoding =
        std::move(std::unique_ptr<zdl::DlSystem::UserBufferEncodingFloat>(
            new zdl::DlSystem::UserBufferEncodingFloat()));
  }

  // create user-backed storage to load input data onto it
  applicationBuffers.emplace(name, std::vector<uint8_t>(bufSize));

  // create SNPE user buffer from the user-backed buffer
  zdl::DlSystem::IUserBufferFactory& ubFactory =
      zdl::SNPE::SNPEFactory::getUserBufferFactory();
  snpeUserBackedBuffers.push_back(
      ubFactory.createUserBuffer(applicationBuffers.at(name).data(), bufSize,
                                 strides, userBufferEncoding.get()));
  if (snpeUserBackedBuffers.back() == nullptr) {
    std::cerr << "Error while creating user buffer." << std::endl;
    std::abort();
  }

  // add the user-backed buffer to the inputMap, which is later on fed to the
  // network for execution
  userBufferMap.add(name, snpeUserBackedBuffers.back().get());
}

void SnpeInference::createInputBufferMap(
    zdl::DlSystem::UserBufferMap& inputMap,
    std::unordered_map<std::string, std::vector<uint8_t>>& applicationBuffers,
    std::vector<std::unique_ptr<zdl::DlSystem::IUserBuffer>>&
        snpeUserBackedBuffers,
    std::unique_ptr<zdl::SNPE::SNPE>& snpe, bool isTfNBuffer, int bitWidth) {
  // get input tensor names of the network that need to be populated
  const auto& inputNamesOpt = snpe->getInputTensorNames();
  if (!inputNamesOpt)
    throw std::runtime_error("Error obtaining input tensor names");
  const zdl::DlSystem::StringList& inputNames = *inputNamesOpt;
  assert(inputNames.size() > 0);

  // create SNPE user buffers for each application storage buffer
  for (const char* name : inputNames) {
    createUserBuffer(inputMap, applicationBuffers, snpeUserBackedBuffers, snpe,
                     name, isTfNBuffer, bitWidth);
  }
}

void SnpeInference::createOutputBufferMap(
    zdl::DlSystem::UserBufferMap& outputMap,
    std::unordered_map<std::string, std::vector<uint8_t>>& applicationBuffers,
    std::vector<std::unique_ptr<zdl::DlSystem::IUserBuffer>>&
        snpeUserBackedBuffers,
    std::unique_ptr<zdl::SNPE::SNPE>& snpe, bool isTfNBuffer, int bitWidth) {
  // get input tensor names of the network that need to be populated
  const auto& outputNamesOpt = snpe->getOutputTensorNames();
  if (!outputNamesOpt)
    throw std::runtime_error("Error obtaining output tensor names");
  const zdl::DlSystem::StringList& outputNames = *outputNamesOpt;

  // create SNPE user buffers for each application storage buffer
  for (const char* name : outputNames) {
    createUserBuffer(outputMap, applicationBuffers, snpeUserBackedBuffers, snpe,
                     name, isTfNBuffer, bitWidth);
  }
}

void SnpeInference::createUserBuffer(
    zdl::DlSystem::UserBufferMap& userBufferMap,
    std::unordered_map<std::string, GLuint>& applicationBuffers,
    std::vector<std::unique_ptr<zdl::DlSystem::IUserBuffer>>&
        snpeUserBackedBuffers,
    std::unique_ptr<zdl::SNPE::SNPE>& snpe, const char* name) {
  // get attributes of buffer by name
  auto bufferAttributesOpt = snpe->getInputOutputBufferAttributes(name);
  if (!bufferAttributesOpt)
    throw std::runtime_error(
        std::string("Error obtaining attributes for input tensor ") + name);
  // calculate the size of buffer required by the input tensor
  const zdl::DlSystem::TensorShape& bufferShape =
      (*bufferAttributesOpt)->getDims();

  // calculate stride based on buffer strides
  // Note: Strides = Number of bytes to advance to the next element in each
  // dimension. For example, if a float tensor of dimension 2x4x3 is tightly
  // packed in a buffer of 96 bytes, then the strides would be (48,12,4)
  std::vector<size_t> strides(bufferShape.rank());
  strides[strides.size() - 1] = sizeof(float);
  size_t stride = strides[strides.size() - 1];
  for (size_t i = bufferShape.rank() - 1; i > 0; i--) {
    stride *= bufferShape[i];
    strides[i - 1] = stride;
  }

  const size_t bufferElementSize = (*bufferAttributesOpt)->getElementSize();
  size_t bufSize = calcSizeFromDims(bufferShape.getDimensions(),
                                    bufferShape.rank(), bufferElementSize);

  // set the buffer encoding type
  zdl::DlSystem::UserBufferEncodingFloat userBufferEncodingFloat;
  zdl::DlSystem::UserBufferSourceGLBuffer userBufferSourceGLBuffer;

  // create user-backed storage to load input data onto it
  applicationBuffers.emplace(name, GLuint(1));

  // create SNPE user buffer from the user-backed buffer
  zdl::DlSystem::IUserBufferFactory& ubFactory =
      zdl::SNPE::SNPEFactory::getUserBufferFactory();
  snpeUserBackedBuffers.push_back(ubFactory.createUserBuffer(
      &applicationBuffers.at(name), bufSize, strides, &userBufferEncodingFloat,
      &userBufferSourceGLBuffer));

  // add the user-backed buffer to the inputMap, which is later on fed to the
  // network for execution
  userBufferMap.add(name, snpeUserBackedBuffers.back().get());
}

void SnpeInference::createInputBufferMap(
    zdl::DlSystem::UserBufferMap& inputMap,
    std::unordered_map<std::string, GLuint>& applicationBuffers,
    std::vector<std::unique_ptr<zdl::DlSystem::IUserBuffer>>&
        snpeUserBackedBuffers,
    std::unique_ptr<zdl::SNPE::SNPE>& snpe) {
  // get input tensor names of the network that need to be populated
  const auto& inputNamesOpt = snpe->getInputTensorNames();
  if (!inputNamesOpt) {
    throw std::runtime_error("Error obtaining input tensor names");
  }
  const zdl::DlSystem::StringList& inputNames = *inputNamesOpt;
  assert(inputNames.size() > 0);

  // create SNPE user buffers for each application storage buffer
  for (const char* name : inputNames) {
    createUserBuffer(inputMap, applicationBuffers, snpeUserBackedBuffers, snpe,
                     name);
  }
}

}  // namespace inference
}  // namespace nndeploy