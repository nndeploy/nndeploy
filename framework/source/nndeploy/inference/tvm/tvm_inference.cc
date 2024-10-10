
#include "nndeploy/inference/tvm/tvm_inference.h"

#include "nndeploy/inference/tvm/tvm_convert.h"

namespace nndeploy {
namespace inference {

TypeInferenceRegister<TypeInferenceCreator<TvmInference>>
    g_tvm_inference_register(base::kInferenceTypeTvm);

TvmInference::TvmInference(base::InferenceType type) : Inference(type) {}

TvmInference::~TvmInference() {}

base::Status TvmInference::reshape(base::ShapeMap &shape_map) {
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
      return base::kStatusCodeErrorInferenceTvm;
    }
  }

  if (flag) {
    NNDEPLOY_LOGE("TVM doesn`t support reshape input");
    return base::kStatusCodeErrorInferenceTvm;
  }
  return base::kStatusCodeOk;
}

base::Status TvmInference::init() {
  base::Status status = base::kStatusCodeOk;

  TvmInferenceParam *tvm_inference_param =
      dynamic_cast<TvmInferenceParam *>(inference_param_);

  if (device::isHostDeviceType(tvm_inference_param->device_type_)) {
    is_share_command_queue_ = true;
  } else {
    is_share_command_queue_ = false;
  }

  std::string module_file = tvm_inference_param->model_value_[0];
  std::string json_file = tvm_inference_param->model_value_[1];
  std::string param_file = tvm_inference_param->model_value_[2];

  mod_handle_ = tvm::runtime::Module::LoadFromFile(module_file.c_str(), "so");

  // 从json中读取model
  std::ifstream json_reader(json_file.c_str());
  if (json_reader.fail()) {
    NNDEPLOY_LOGE("Failed to open tvm json file: %s", json_file.c_str());
  }
  json_reader.seekg(0, std::ios_base::end);
  std::size_t json_size = json_reader.tellg();
  json_reader.seekg(0, std::ios_base::beg);
  std::string json_data;
  json_data.reserve(json_size);
  json_reader.read((char *)json_data.c_str(), json_size);
  json_reader.close();

  // 创建graph
  auto f_handle = tvm::runtime::Registry::Get("tvm.graph_executor.create");
  graph_handle_ = (*f_handle)(
      json_data, mod_handle_,
      static_cast<int>(
          TvmConvert::convertFromDeviceType(tvm_inference_param->device_type_)),
      0);

  // 读取params
  std::ifstream params_reader(param_file.c_str(), std::ios::binary);
  if (params_reader.fail()) {
    NNDEPLOY_LOGE("Failed to open tvm params file: %s", param_file.c_str());
  }
  params_reader.seekg(0, std::ios_base::end);
  std::size_t param_size = params_reader.tellg();
  params_reader.seekg(0, std::ios_base::beg);
  std::vector<char> param_data(param_size / sizeof(char));
  params_reader.read((char *)&param_data[0], param_size);
  params_reader.close();

  // 载入参数
  TVMByteArray params_arr;
  params_arr.data = (char *)&param_data[0];
  params_arr.size = param_size;
  graph_handle_.GetFunction("load_params")(params_arr);

  // 从tvm runtime module中获取运行需要的输入、输出元信息
  mInfo_.n_inputs = graph_handle_.GetFunction("get_num_inputs")();
  mInfo_.n_outputs = graph_handle_.GetFunction("get_num_outputs")();

  tvm::runtime::Map<tvm::runtime::String, tvm::runtime::ObjectRef>
      tvm_input_info = graph_handle_.GetFunction("get_input_info")();
  auto shape_info = tvm::runtime::GetRef<
      tvm::runtime::Map<tvm::runtime::String, tvm::runtime::ObjectRef>>(
      tvm_input_info["shape"].as<tvm::runtime::MapNode>());
  auto dtype_info = tvm::runtime::GetRef<
      tvm::runtime::Map<tvm::runtime::String, tvm::runtime::ObjectRef>>(
      tvm_input_info["dtype"].as<tvm::runtime::MapNode>());
  for (const auto &kv : shape_info) {
    auto stuple = tvm::runtime::GetRef<tvm::runtime::ShapeTuple>(
        kv.second.as<tvm::runtime::ShapeTupleObj>());
    std::vector<int64_t> vshape;
    vshape.assign(stuple.begin(), stuple.end());
    auto dtype = tvm::runtime::GetRef<tvm::runtime::String>(
        dtype_info[kv.first].as<tvm::runtime::StringObj>());
    std::pair<std::vector<int64_t>, std::string> value =
        std::make_pair(vshape, dtype);
    mInfo_.input_info.insert({kv.first, value});
  }

  tvm_input_info = graph_handle_.GetFunction("get_output_info")();
  shape_info = tvm::runtime::GetRef<
      tvm::runtime::Map<tvm::runtime::String, tvm::runtime::ObjectRef>>(
      tvm_input_info["shape"].as<tvm::runtime::MapNode>());
  dtype_info = tvm::runtime::GetRef<
      tvm::runtime::Map<tvm::runtime::String, tvm::runtime::ObjectRef>>(
      tvm_input_info["dtype"].as<tvm::runtime::MapNode>());
  for (const auto &kv : shape_info) {
    auto stuple = tvm::runtime::GetRef<tvm::runtime::ShapeTuple>(
        kv.second.as<tvm::runtime::ShapeTupleObj>());
    std::vector<int64_t> vshape;
    vshape.assign(stuple.begin(), stuple.end());
    auto dtype = tvm::runtime::GetRef<tvm::runtime::String>(
        dtype_info[kv.first].as<tvm::runtime::StringObj>());
    std::pair<std::vector<int64_t>, std::string> value =
        std::make_pair(vshape, dtype);
    mInfo_.output_info.insert({kv.first, value});
  }

  status = allocateInputOutputTensor();
  NNDEPLOY_RETURN_ON_NEQ(status, base::kStatusCodeOk,
                         "allocateInputOutputTensor failed!!\n");

  return status;
}

base::Status TvmInference::deinit() {
  base::Status status = deallocateInputOutputTensor();
  NNDEPLOY_RETURN_ON_NEQ(status, base::kStatusCodeOk,
                         "deallocateInputOutputTensor failed!!\n");
  return base::kStatusCodeOk;
}

base::Status TvmInference::allocateInputOutputTensor() {
  for (auto &elem : mInfo_.input_info) {
    tvm::runtime::NDArray input =
        graph_handle_.GetFunction("get_input")(elem.first);
    device::Tensor *input_tensor =
        TvmConvert::convertToTensor(input, elem.first);
    input_tensors_.insert({elem.first, input_tensor});
  }

  for (auto &elem : mInfo_.output_info) {
    tvm::runtime::NDArray output =
        graph_handle_.GetFunction("get_output")(elem.first);
    device::Tensor *output_tensor =
        TvmConvert::convertToTensor(output, elem.first);

    output_tensors_.insert({elem.first, output_tensor});
  }
  return base::kStatusCodeOk;
}

base::Status TvmInference::deallocateInputOutputTensor() {
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

size_t TvmInference::getMemSize(const tvm::runtime::NDArray &narr) {
  size_t size = 1;
  for (int i = 0; i < narr->ndim; ++i) {
    size *= static_cast<size_t>(narr->shape[i]);
  }
  size *= (narr->dtype.bits * narr->dtype.lanes + 7) / 8;
  return size;
}

base::Status TvmInference::run() {
  // 设置输入
  for (auto input : external_input_tensors_) {
    tvm::runtime::NDArray tvm_array =
        graph_handle_.GetFunction("get_input")(input.first);
    auto ssize = getMemSize(tvm_array);
    tvm_array.CopyFromBytes(input.second->getData(),
                            ssize);  // TODO: 有没有zero copy的实现
  }

  graph_handle_.GetFunction("run")();

  return base::kStatusCodeOk;
}

device::Tensor *TvmInference::getOutputTensorAfterRun(
    const std::string &name, base::DeviceType device_type, bool is_copy,
    base::DataFormat data_format) {
  device::Device *device = device::getDevice(device_type);
  tvm::runtime::NDArray tvm_tensor =
      graph_handle_.GetFunction("get_output")(name);
  device::Tensor *internal_tensor =
      TvmConvert::convertToTensor(tvm_tensor, name);
  device::TensorDesc desc = internal_tensor->getDesc();
  device::Tensor *output_tensor = nullptr;
  bool flag = is_copy || (internal_tensor->getDevice() != device);

  if (flag) {
    output_tensor = new device::Tensor(device, desc, name);
    internal_tensor->getBuffer()->copyTo(output_tensor->getBuffer());

  } else {
    output_tensor = internal_tensor;
  }
  return output_tensor;
}

}  // namespace inference
}  // namespace nndeploy