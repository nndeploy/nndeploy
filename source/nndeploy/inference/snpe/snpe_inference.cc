#include "nndeploy/inference/snpe/snpe_inference.h"

namespace nndeploy
{
namespace inference
{

TypeInferenceRegister<TypeInferenceCreator<SnpeInference>>
    g_snpe_inference_register(base::kInferenceTypeSnpe);

SnpeInference::SnpeInference(base::InferenceType type) : Inference(type) {}
SnpeInference::~SnpeInference() {}


base::Status SnpeInference::init()
{
    base::Status status = base::kStatusCodeOk;

    SnpeInferenceParam *snpe_inference_param =
        dynamic_cast<SnpeInferenceParam *>(inference_param_);

        base::Status status = base::kStatusCodeOk;

    zdl::DlSystem::Runtime_t runtime;
    std::string snpe_runtime = snpe_inference_param->snpe_runtime_;
    NNDEPLOY_LOGE("snpe_perf_mode is : [%s].\n", snpe_runtime.c_str());
    if (snpe_runtime == "cpu")
    {
        runtime = zdl::DlSystem::Runtime_t::CPU;
    }
    else if (snpe_runtime == "gpu")
    {
        runtime = zdl::DlSystem::Runtime_t::GPU;
    }
    else if (snpe_runtime == "dsp")
    {
        runtime = zdl::DlSystem::Runtime_t::DSP;
    }
    else if (snpe_runtime == "aip")
    {
        runtime = zdl::DlSystem::Runtime_t::AIP_FIXED8_TF;
    }
    else
    {
        NNDEPLOY_RETURN_ON_NEQ("SNPE runtime option[%s] is not valid!\
                Defaulting to the CPU runtime.\n", snpe_runtime.c_str());
        runtime = zdl::DlSystem::Runtime_t::CPU;
    }
    if (!zdl::SNPE::SNPEFactory::isRuntimeAvailable(runtime))
    {
        NNDEPLOY_RETURN_ON_NEQ("Selected runtime[%d] not available!\
            Please check your environment and runtime setting in .json!\n", static_cast<int>(runtime));
        std::abort();
    }

    zdl::DlSystem::PerformanceProfile_t perf_mode;
    int32_t snpe_perf_mode = snpe_inference_param->snpe_perf_mode_;
    NNDEPLOY_LOGE("snpe_perf_mode is : [%d].\n", snpe_perf_mode);
    switch (snpe_perf_mode)
    {
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
            perf_mode = zdl::DlSystem::PerformanceProfile_t::SUSTAINED_HIGH_PERFORMANCE;
            break;
        case 5:
            perf_mode = zdl::DlSystem::PerformanceProfile_t::BURST;
            break;
        default:
            NNDEPLOY_RETURN_ON_EQ("SNPE performance mode[%d] is not valid!\n", snpe_perf_mode);
            std::abort();
    }

    zdl::DlSystem::ProfilingLevel_t profiling_level;
    int32_t snpe_profiling_level = snpe_inference_param->snpe_profiling_level_;
    NNDEPLOY_LOGE("snpe_profiling_level is : [%d].\n", snpe_profiling_level);
    switch (snpe_profiling_level)
    {
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
            NNDEPLOY_RETURN_ON_NEQ("SNPE profiling level[%d] is not valid!\n", snpe_profiling_level);
            std::abort();
    }

    int32_t buffer_type = snpe_inference_param->snpe_buffer_type_;
    int32_t bit_width;
    switch (buffer_type)
    {
    case 0:
        buffer_type = USERBUFFER_FLOAT;
        bit_width = 32;
        break;
    case 1:
        buffer_type = USERBUFFER_TF8;
        bit_width = 8;
        break;
    case 2:
        buffer_type = ITENSOR;
        bit_width = 32; // or 8
        break;
    case 3:
        buffer_type = USERBUFFER_TF16;
        bit_width = 16;
        break;
    default:
        NNDEPLOY_RETURN_ON_EQ("The buffer type is not supported.\n");
        break;
    }
    bool use_user_supplied_buffers = (buffer_type == USERBUFFER_FLOAT ||
                                      buffer_type == USERBUFFER_TF8 ||
                                      buffer_type == USERBUFFER_TF16 ||
                                      buffer_type == ITENSOR);

    std::string modelfile_path = snpe_inference_param->model_value_[0];
    std::ifstream dlcFile(modelfile_path);
    if (!dlcFile)
    {
        NNDEPLOY_LOGE("DLC file[%s] not valid. Please ensure that you\
                have provided a valid dlc file!\n", modelfile_path.c_str());
        std::abort();
    }

    std::unique_ptr<zdl::DlContainer::IDlContainer> container =
            zdl::DlContainer::IDlContainer::open(zdl::DlSystem::String(modelfile_path.c_str()));
    if (container == nullptr)
    {
        NNDEPLOY_RETURN_ON_EQ("Error while opening the container file[%s].\n", modelfile_path.c_str());
        return nullptr;
    }

    zdl::DlSystem::UDLFactoryFunc udl_func = UdlExample::MyUDLFactory;
    zdl::DlSystem::UDLBundle udl_bundle;
    udl_bundle.cookie = (void*)0xdeadbeaf, udl_bundle.func = udl_func; // 0xdeadbeaf to test cookie
    zdl::DlSystem::PlatformConfig platform_config;
    bool usingInitCaching = false;

    zdl::SNPE::SNPEBuilder snpe_builder(container.get());

    zdl::DlSystem::StringList outputLayerNames = {};
    std::vector<std::string> output_layer_names = snpe_inference_param->output_layer_names_;
    for (int i = 0; i < output_layer_names.size(); i++)
    {
        outputLayerNames.append(output_layer_names[i].c_str());
    }

    if (outputLayerNames.size() != 0)
    {
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

    if (snpe_ == nullptr)
    {
        NNDEPLOY_RETURN_ON_EQ("Error while building SNPE object.\n");
        return nullptr;
    }

    if (buffer_type == USERBUFFER_FLOAT)
    {
        createInputBufferMap(*input_map_, application_input_buffers_, snpe_user_input_buffers_, snpe_, false, bit_width);
        createOutputBufferMap(*output_map_, application_output_buffers_, snpe_user_output_buffers_, snpe_, false, bit_width);
    }
    else if (buffer_type == USERBUFFER_TF8 || buffer_type == USERBUFFER_TF16)
    {
        createInputBufferMap(*input_map_, application_input_buffers_, snpe_user_input_buffers_, snpe_, true, bit_width);
        createOutputBufferMap(*output_map_, application_output_buffers_, snpe_user_output_buffers_, snpe_, true, bit_width);
    }
    else if (buffer_type == ITENSOR)
    {
        createITensors(&inputTensor_, snpe_, 1);
    }

    status = allocateInputOutputTensor();

    return status;
}

base::Status SnpeInference::deinit()
{
    base::Status status = deallocateInputOutputTensor();
    NNDEPLOY_RETURN_ON_NEQ(status, base::kStatusCodeOk,
                         "deallocateInputOutputTensor failed!!\n");

    return base::kStatusCodeOk;
}

base::Status SnpeInference::run()
{
    bool execStatus = snpe_->execute(input_map_, output_map_);
    if (execStatus != true)
    {
        std::cout << "Error while executing the network! Error info: "
                << zdl::DlSystem::getLastErrorString() << std::endl;

        return kStatusCodeErrorInferenceSnpe;
    }

    return base::kStatusCodeOk;
}

device::Tensor *SnpeInference::getOutputTensorAfterRun(
            const std::string &name, base::DeviceType device_type, bool is_copy,
            base::DataFormat data_format = base::kDataFormatAuto)
{
    device::Device *device = device::getDevice(device_type);
    device::Tensor *internal_tensor = output_tensors_[name];
    device::TensorDesc desc = internal_tensor->getDesc();
    bool flag = is_copy || (internal_tensor->getDevice() != device);
    if (flag)
    {
        device::Tensor *output_tensor = new device::Tensor(device, desc, name);
        internal_tensor->getBuffer()->copyTo(output_tensor->getBuffer());
        return output_tensor;
    }
    else
    {
        device::Tensor *output_tensor = new device::Tensor(desc, internal_tensor->getBuffer(), name);
        return output_tensor;
    }
}

base::Status SnpeInference::allocateInputOutputTensor()
{
    /* debug info: input tensor name, size; output tensor name, size */
    const zdl::DlSystem::Optional<zdl::DlSystem::StringList> &input_tensor_names_opt =
            snpe_->getInputTensorNames();
    const zdl::DlSystem::Optional<zdl::DlSystem::StringList> &output_tensor_names_opt =
            snpe_->getOutputTensorNames();
    const zdl::DlSystem::Optional<zdl::DlSystem::StringList> &output_layer_names_opt =
            snpe_->getOutputLayerNames();
    const zdl::DlSystem::StringList& input_tensor_names = *input_tensor_names_opt;
    const zdl::DlSystem::StringList& output_tensor_names = *output_tensor_names_opt;
    const zdl::DlSystem::StringList& output_layer_names = *output_layer_names_opt;

    device::Device *device = nullptr;
    if (device::isHostDeviceType(inference_param_->device_type_))
    {
        device = device::getDevice(inference_param_->device_type_);
    }

    for (const char* name : input_tensor_names)
    {
        std::cout << "input tensor[" << name << "] dims: [ ";
        const zdl::DlSystem::Optional<zdl::DlSystem::TensorShape> &tensorShapeOpt =
                snpe_->getInputDimensions(name);
        const zdl::DlSystem::TensorShape& tensorShape = *tensorShapeOpt;
        for (size_t i = 0; i < tensorShape.rank(); i++)
        {
            std::cout << tensorShape[i] << "  ";
        }
        std::cout << "]" << std::endl;

        zdl::DlSystem::Dimension &dims = tensorShape.getDimensions();
        size_t rank = tensorShape.rank();
        base::IntVector shape = SnpeConvert::convertToShape(dims, rank);
        base::DataType data_type = SnpeConvert::convertToDataType(buffer_type);
        base::DataFormat data_format = SnpeConvert::convertToDataFormat();
        device::TensorDesc desc;
        desc.data_type_ = data_type;
        desc.data_format_ = data_format;
        desc.shape_ = shape;
        desc.stride_ = base::SizeVector();

        device::Tensor *input_tensor = nullptr;
        if (device == nullptr)
        {
            input_tensor = new device::Tensor(desc, name);
        }
        else
        {
            void* data_ptr = reinterpret_cast<void*>(&application_input_buffers_.at(name.c_str())[0]);
            base::IntVector memory_config = base::IntVector();
            input_tensor = new device::Tensor(device, desc, data_ptr, name, memory_config);
        }
        
        input_tensors_.insert({name, input_tensor});
    }
    for (const char* name : output_tensor_names)
    {
        std::cout << "output tensor[" << name << "] dims: [ ";
        auto bufferAttributesOpt = snpe_->getInputOutputBufferAttributes(name);
        const zdl::DlSystem::TensorShape& tensorShape = (*bufferAttributesOpt)->getDims();
        for (size_t i = 0; i < tensorShape.rank(); i++)
        {
            std::cout << tensorShape[i] << "  ";
        }
        std::cout << "]" << std::endl;

        zdl::DlSystem::Dimension &dims = tensorShape.getDimensions();
        size_t rank = tensorShape.rank();
        base::IntVector shape = SnpeConvert::convertToShape(dims, rank);
        base::DataType data_type = SnpeConvert::convertToDataType(buffer_type);
        base::DataFormat data_format = SnpeConvert::convertToDataFormat();
        device::TensorDesc desc;
        desc.data_type_ = data_type;
        desc.data_format_ = data_format;
        desc.shape_ = shape;
        desc.stride_ = base::SizeVector();

        device::Tensor *output_tensor = nullptr;
        if (device == nullptr)
        {
            output_tensor = new device::Tensor(desc, name);
        }
        else
        {
            void* data_ptr = reinterpret_cast<void*>(&application_output_buffers_.at(name.c_str())[0]);
            base::IntVector memory_config = base::IntVector();
            output_tensor = new device::Tensor(device, desc, data_ptr, name, memory_config);
        }

        output_tensors_.insert({name, output_tensor});
    }
    for (const char* name : output_layer_names)
    {
        std::cout << "output layer[" << name << "]" << std::endl;
    }
}

base::Status SnpeInference::deallocateInputOutputTensor()
{
    for (auto iter : input_tensors_)
    {
        delete iter.second;
    }
    input_tensors_.clear();

    for (auto iter : output_tensors_)
    {
        delete iter.second;
    }
    output_tensors_.clear();

    return base::kStatusCodeOk;
}

size_t SnpeInference::calcSizeFromDims(const zdl::DlSystem::Dimension *dims,
                        size_t rank, size_t elementSize)
{
    if (rank == 0) return 0;
    size_t size = elementSize;

    while (rank--)
    {
        (*dims == 0) ? size *= resizable_dim : size *= *dims;
        dims++;
    }

    return size;
}

void SnpeInference::setResizableDim(size_t resizableDim)
{
    resizable_dim = resizableDim;
}

size_t SnpeInference::getResizableDim()
{
    return resizable_dim;
}

void SnpeInference::createUserBuffer(zdl::DlSystem::UserBufferMap& userBufferMap,
                      std::unordered_map<std::string, std::vector<uint8_t>>& applicationBuffers,
                      std::vector<std::unique_ptr<zdl::DlSystem::IUserBuffer>>& snpeUserBackedBuffers,
                      std::unique_ptr<zdl::SNPE::SNPE>& snpe,
                      const char * name,
                      const bool isTfNBuffer,
                      int bitWidth)
{
    // get attributes of buffer by name
    auto bufferAttributesOpt = snpe->getInputOutputBufferAttributes(name);
    if (!bufferAttributesOpt)
    {
        throw std::runtime_error(std::string("Error obtaining attributes for input tensor ") + name);
    }

    // calculate the size of buffer required by the input tensor
    const zdl::DlSystem::TensorShape& bufferShape = (*bufferAttributesOpt)->getDims();

    size_t bufferElementSize = 0;
    if (isTfNBuffer)
    {
        bufferElementSize = bitWidth / 8;
    }
    else
    {
        bufferElementSize = sizeof(float);
    }

    // Calculate the stride based on buffer strides.
    // Note: Strides = Number of bytes to advance to the next element in each dimension.
    // For example, if a float tensor of dimension 2x4x3 is tightly packed in a buffer of 96 bytes, then the strides would be (48,12,4)
    // Note: Buffer stride is usually known and does not need to be calculated.
    std::vector<size_t> strides(bufferShape.rank());
    strides[strides.size() - 1] = bufferElementSize;
    size_t stride = strides[strides.size() - 1];
    for (size_t i = bufferShape.rank() - 1; i > 0; i--)
    {
        (bufferShape[i] == 0) ? stride *= getResizableDim() : stride *= bufferShape[i];
        strides[i-1] = stride;
    }

    size_t bufSize = calcSizeFromDims(bufferShape.getDimensions(), bufferShape.rank(), bufferElementSize);

    // set the buffer encoding type
    std::unique_ptr<zdl::DlSystem::UserBufferEncoding> userBufferEncoding;
    if (isTfNBuffer)
    {
        userBufferEncoding = std::unique_ptr<zdl::DlSystem::UserBufferEncodingTfN>(
            new zdl::DlSystem::UserBufferEncodingTfN(0,1.0, bitWidth));
    }
    else
    {
        userBufferEncoding = std::unique_ptr<zdl::DlSystem::UserBufferEncodingFloat>(
            new zdl::DlSystem::UserBufferEncodingFloat());
    }

    // create user-backed storage to load input data onto it
    applicationBuffers.emplace(name, std::vector<uint8_t>(bufSize));

    // create SNPE user buffer from the user-backed buffer
    zdl::DlSystem::IUserBufferFactory& ubFactory = zdl::SNPE::SNPEFactory::getUserBufferFactory();
    snpeUserBackedBuffers.push_back(ubFactory.createUserBuffer(applicationBuffers.at(name).data(),
                                                                bufSize,
                                                                strides,
                                                                userBufferEncoding.get()));
    if (snpeUserBackedBuffers.back() == nullptr)
    {
        std::cerr << "Error while creating user buffer." << std::endl;
        std::abort();
    }
    // add the user-backed buffer to the inputMap, which is later on fed to the network for execution
    userBufferMap.add(name, snpeUserBackedBuffers.back().get());
}

void SnpeInference::createInputBufferMap(zdl::DlSystem::UserBufferMap& inputMap,
                          std::unordered_map<std::string, std::vector<uint8_t>>& applicationBuffers,
                          std::vector<std::unique_ptr<zdl::DlSystem::IUserBuffer>>& snpeUserBackedBuffers,
                          std::unique_ptr<zdl::SNPE::SNPE>& snpe,
                          bool isTfNBuffer,
                          int bitWidth)
{
    // get input tensor names of the network that need to be populated
    const auto& inputNamesOpt = snpe->getInputTensorNames();
    if (!inputNamesOpt) throw std::runtime_error("Error obtaining input tensor names");
    const zdl::DlSystem::StringList& inputNames = *inputNamesOpt;
    assert(inputNames.size() > 0);

    // create SNPE user buffers for each application storage buffer
    for (const char *name : inputNames)
    {
        createUserBuffer(inputMap, applicationBuffers, snpeUserBackedBuffers, snpe, name, isTfNBuffer, bitWidth);
    }
}

void SnpeInference::createOutputBufferMap(zdl::DlSystem::UserBufferMap& outputMap,
                           std::unordered_map<std::string, std::vector<uint8_t>>& applicationBuffers,
                           std::vector<std::unique_ptr<zdl::DlSystem::IUserBuffer>>& snpeUserBackedBuffers,
                           std::unique_ptr<zdl::SNPE::SNPE>& snpe,
                           bool isTfNBuffer,
                           int bitWidth)
{
    // get input tensor names of the network that need to be populated
    const auto& outputNamesOpt = snpe->getOutputTensorNames();
    if (!outputNamesOpt) throw std::runtime_error("Error obtaining output tensor names");
    const zdl::DlSystem::StringList& outputNames = *outputNamesOpt;

    // create SNPE user buffers for each application storage buffer
    for (const char *name : outputNames)
    {
        createUserBuffer(outputMap, applicationBuffers, snpeUserBackedBuffers, snpe, name, isTfNBuffer, bitWidth);
    }
}

void SnpeInference::createUserBuffer(zdl::DlSystem::UserBufferMap& userBufferMap,
                      std::unordered_map<std::string, GLuint>& applicationBuffers,
                      std::vector<std::unique_ptr<zdl::DlSystem::IUserBuffer>>& snpeUserBackedBuffers,
                      std::unique_ptr<zdl::SNPE::SNPE>& snpe,
                      const char * name)
{
    // get attributes of buffer by name
    auto bufferAttributesOpt = snpe->getInputOutputBufferAttributes(name);
    if (!bufferAttributesOpt) throw std::runtime_error(std::string("Error obtaining attributes for input tensor ") + name);
    // calculate the size of buffer required by the input tensor
    const zdl::DlSystem::TensorShape& bufferShape = (*bufferAttributesOpt)->getDims();

    // calculate stride based on buffer strides
    // Note: Strides = Number of bytes to advance to the next element in each dimension.
    // For example, if a float tensor of dimension 2x4x3 is tightly packed in a buffer of 96 bytes, then the strides would be (48,12,4)
    std::vector<size_t> strides(bufferShape.rank());
    strides[strides.size() - 1] = sizeof(float);
    size_t stride = strides[strides.size() - 1];
    for (size_t i = bufferShape.rank() - 1; i > 0; i--)
    {
            stride *= bufferShape[i];
            strides[i-1] = stride;
    }

    const size_t bufferElementSize = (*bufferAttributesOpt)->getElementSize();
    size_t bufSize = calcSizeFromDims(bufferShape.getDimensions(), bufferShape.rank(), bufferElementSize);

    // set the buffer encoding type
    zdl::DlSystem::UserBufferEncodingFloat userBufferEncodingFloat;
    zdl::DlSystem::UserBufferSourceGLBuffer userBufferSourceGLBuffer;

    // create user-backed storage to load input data onto it
    applicationBuffers.emplace(name, GLuint(1));

    // create SNPE user buffer from the user-backed buffer
    zdl::DlSystem::IUserBufferFactory& ubFactory = zdl::SNPE::SNPEFactory::getUserBufferFactory();
    snpeUserBackedBuffers.push_back(ubFactory.createUserBuffer(&applicationBuffers.at(name),
                                                                bufSize,
                                                                strides,
                                                                &userBufferEncodingFloat,
                                                                &userBufferSourceGLBuffer));

    // add the user-backed buffer to the inputMap, which is later on fed to the network for execution
    userBufferMap.add(name, snpeUserBackedBuffers.back().get());
}

void SnpeInference::createInputBufferMap(zdl::DlSystem::UserBufferMap& inputMap,
                          std::unordered_map<std::string, GLuint>& applicationBuffers,
                          std::vector<std::unique_ptr<zdl::DlSystem::IUserBuffer>>& snpeUserBackedBuffers,
                          std::unique_ptr<zdl::SNPE::SNPE>& snpe)
{
    // get input tensor names of the network that need to be populated
    const auto& inputNamesOpt = snpe->getInputTensorNames();
    if (!inputNamesOpt)
    {
        throw std::runtime_error("Error obtaining input tensor names");
    }
    const zdl::DlSystem::StringList& inputNames = *inputNamesOpt;
    assert(inputNames.size() > 0);

    // create SNPE user buffers for each application storage buffer
    for (const char *name : inputNames)
    {
        createUserBuffer(inputMap, applicationBuffers, snpeUserBackedBuffers, snpe, name);
    }
}

void SnpeInference::createITensors(std::unique_ptr<zdl::DlSystem::ITensor>* inputs,
                        std::unique_ptr<zdl::SNPE::SNPE>& snpe,
                        const size_t inputSize)
{
    const auto &inputTensorNamesRef = snpe->getInputTensorNames();
    if (!inputTensorNamesRef)
        throw std::runtime_error("Error obtaining Input tensor names!");
    const auto &inputTensorNames = *inputTensorNamesRef;
    if (inputTensorNames.size() != inputSize)
        throw std::runtime_error("Input tensor's size is not equal to inputSize!");

    for (size_t i = 0; i < inputTensorNames.size(); i++)
    {
        const auto &inputDimsRef = snpe->getInputDimensions(inputTensorNames.at(i));
        const auto &inputDims = *inputDimsRef;
        inputs[i] = zdl::SNPE::SNPEFactory::getTensorFactory().createTensor(inputDims);
    }
}

} // namespace inference
} // namespace nndeploy