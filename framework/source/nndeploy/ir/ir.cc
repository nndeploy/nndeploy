
#include "nndeploy/ir/ir.h"

#include <memory>

#include "nndeploy/base/macro.h"
#include "nndeploy/base/status.h"
#include "nndeploy/device/tensor.h"
#include "safetensors.hh"

namespace nndeploy {
namespace ir {

OpDesc::OpDesc() {}
OpDesc::OpDesc(OpType op_type) : op_type_(op_type) {
  op_param_ = createOpParam(op_type);
}
OpDesc::OpDesc(const std::string &name, OpType op_type)
    : name_(name), op_type_(op_type) {
  op_param_ = createOpParam(op_type);
}
OpDesc::OpDesc(const std::string &name, OpType op_type,
               std::shared_ptr<base::Param> op_param)
    : name_(name), op_type_(op_type) {
  if (op_param != nullptr) {
    op_param_ = op_param->copy();
  } else {
    op_param_ = createOpParam(op_type);
  }
}
OpDesc::OpDesc(const std::string &name, OpType op_type,
               std::initializer_list<std::string> inputs,
               std::initializer_list<std::string> outputs)
    : name_(name), op_type_(op_type), inputs_(inputs), outputs_(outputs) {
  op_param_ = createOpParam(op_type);
}
OpDesc::OpDesc(const std::string &name, OpType op_type,
               std::initializer_list<std::string> inputs,
               std::initializer_list<std::string> outputs,
               std::shared_ptr<base::Param> op_param)
    : name_(name), op_type_(op_type), inputs_(inputs), outputs_(outputs) {
  if (op_param != nullptr) {
    op_param_ = op_param->copy();
  } else {
    op_param_ = createOpParam(op_type);
  }
}
OpDesc::OpDesc(const std::string &name, OpType op_type,
               std::vector<std::string> &inputs,
               std::vector<std::string> &outputs)
    : name_(name), op_type_(op_type), inputs_(inputs), outputs_(outputs) {
  op_param_ = createOpParam(op_type);
}
OpDesc::OpDesc(const std::string &name, OpType op_type,
               std::vector<std::string> &inputs,
               std::vector<std::string> &outputs,
               std::shared_ptr<base::Param> op_param)
    : name_(name), op_type_(op_type), inputs_(inputs), outputs_(outputs) {
  if (op_param != nullptr) {
    op_param_ = op_param->copy();
  } else {
    op_param_ = createOpParam(op_type);
  }
}

OpDesc::~OpDesc() {}

/**
 * @brief
 *
 * @param stream
 * @return base::Status
 * @note
 * name_: model.0.conv
 * op_type_: kOpTypeConv
 * inputs_: [images, model.0.conv.weight, model.0.conv.bias]
 * outputs_: [model.0.conv_output_0]
 * op_param_: {
 *   "auto_pad_": "NOTSET"
 *   "dilations_": [1,1]
 *   "group_": 1
 *   "kernel_shape_": [3,3]
 *   "pads_": [1,1,1,1]
 *   "strides_": [2,2]
 *   "is_fusion_op_": 0
 *   "activate_op_": "kOpTypeRelu"
 * }
 */
base::Status OpDesc::serialize(
    rapidjson::Value &json,
    rapidjson::Document::AllocatorType &allocator) const {
  // 写入算子名称
  json.AddMember("name_", rapidjson::Value(name_.c_str(), allocator),
                 allocator);
  // 写入算子类型
  json.AddMember("op_type_",
                 rapidjson::Value(opTypeToString(op_type_).c_str(), allocator),
                 allocator);
  // 写入输入
  rapidjson::Value inputs(rapidjson::kArrayType);
  for (const auto &input : inputs_) {
    inputs.PushBack(rapidjson::Value(input.c_str(), allocator), allocator);
  }
  json.AddMember("inputs_", inputs, allocator);
  // 写入输出
  rapidjson::Value outputs(rapidjson::kArrayType);
  for (const auto &output : outputs_) {
    outputs.PushBack(rapidjson::Value(output.c_str(), allocator), allocator);
  }
  json.AddMember("outputs_", outputs, allocator);
  // 写入参数
  if (op_param_ != nullptr) {
    rapidjson::Value param_json(rapidjson::kObjectType);
    op_param_->serialize(param_json, allocator);
    json.AddMember("op_param_", param_json, allocator);
  }

  return base::kStatusCodeOk;
}
/**
 * @brief
 *
 * @param stream
 * @return base::Status
 * @note
 * name_: model.0.conv
 * op_type_: kOpTypeConv
 * inputs_: [images, model.0.conv.weight, model.0.conv.bias]
 * outputs_: [model.0.conv_output_0]
 * op_param_: {
 *   "auto_pad_": "NOTSET"
 *   "dilations_": [1,1]
 *   "group_": 1
 *   "kernel_shape_": [3,3]
 *   "pads_": [1,1,1,1]
 *   "strides_": [2,2]
 *   "is_fusion_op_": 0
 *   "activate_op_": "kOpTypeRelu"
 * }
 */
base::Status OpDesc::deserialize(rapidjson::Value &json) {
  // 读取算子名称
  if (!json.HasMember("name_") || !json["name_"].IsString()) {
    return base::kStatusCodeErrorInvalidValue;
  }
  name_ = json["name_"].GetString();

  // 读取算子类型
  if (!json.HasMember("op_type_") || !json["op_type_"].IsString()) {
    return base::kStatusCodeErrorInvalidValue;
  }
  op_type_ = stringToOpType(json["op_type_"].GetString());

  // 读取输入
  if (!json.HasMember("inputs_") || !json["inputs_"].IsArray()) {
    return base::kStatusCodeErrorInvalidValue;
  }
  inputs_.clear();
  const rapidjson::Value &inputs = json["inputs_"];
  for (rapidjson::SizeType i = 0; i < inputs.Size(); i++) {
    if (!inputs[i].IsString()) {
      return base::kStatusCodeErrorInvalidValue;
    }
    inputs_.push_back(inputs[i].GetString());
  }

  // 读取输出
  if (!json.HasMember("outputs_") || !json["outputs_"].IsArray()) {
    return base::kStatusCodeErrorInvalidValue;
  }
  outputs_.clear();
  const rapidjson::Value &outputs = json["outputs_"];
  for (rapidjson::SizeType i = 0; i < outputs.Size(); i++) {
    if (!outputs[i].IsString()) {
      return base::kStatusCodeErrorInvalidValue;
    }
    outputs_.push_back(outputs[i].GetString());
  }

  // 读取参数
  if (json.HasMember("op_param_") && json["op_param_"].IsObject()) {
    op_param_ = createOpParam(op_type_);
    if (op_param_ != nullptr) {
      rapidjson::Value &param_json = json["op_param_"];
      base::Status status = op_param_->deserialize(param_json);
      NNDEPLOY_RETURN_VALUE_ON_NEQ(status, base::kStatusCodeOk, status,
                                   "op_param_->deserialize failed!\n");
    }
  }

  return base::kStatusCodeOk;
}

ValueDesc::ValueDesc() { data_type_ = base::dataTypeOf<float>(); }
ValueDesc::ValueDesc(const std::string &name) : name_(name) {
  data_type_ = base::dataTypeOf<float>();
}
ValueDesc::ValueDesc(const std::string &name, base::DataType data_type)
    : name_(name), data_type_(data_type) {}
ValueDesc::ValueDesc(const std::string &name, base::DataType data_type,
                     base::IntVector shape)
    : name_(name), data_type_(data_type), shape_(shape) {}

ValueDesc::~ValueDesc() {
  name_.clear();
  shape_.clear();
}

/**
 * @brief
 *
 * @param stream
 * @return base::Status
 * @note
 * name_: model.0.conv_output_0
 * data_type_: dataTypeToString(data_type_)
 * shape_: [1, 32, 16, 16]
 */
base::Status ValueDesc::serialize(
    rapidjson::Value &json,
    rapidjson::Document::AllocatorType &allocator) const {
  // 序列化名称
  json.AddMember("name_", rapidjson::Value(name_.c_str(), allocator),
                 allocator);

  // 序列化数据类型
  if (data_type_.code_ != base::kDataTypeCodeNotSupport) {
    std::string data_type_str = base::dataTypeToString(data_type_);
    json.AddMember("data_type_",
                   rapidjson::Value(data_type_str.c_str(), allocator),
                   allocator);
  }

  // 序列化形状
  if (!shape_.empty()) {
    rapidjson::Value shape_array(rapidjson::kArrayType);
    for (auto dim : shape_) {
      shape_array.PushBack(dim, allocator);
    }
    json.AddMember("shape_", shape_array, allocator);
  }

  return base::kStatusCodeOk;
}
/**
 * @brief
 *
 * @param stream
 * @return base::Status
 * @note
 * name_: model.0.conv_output_0
 * data_type_: dataTypeToString(data_type_)
 * shape_: [1, 32, 16, 16]
 */
base::Status ValueDesc::deserialize(rapidjson::Value &json) {
  // 解析名称
  if (json.HasMember("name_")) {
    name_ = json["name_"].GetString();
  }

  // 解析数据类型
  if (json.HasMember("data_type_")) {
    std::string data_type_str = json["data_type_"].GetString();
    data_type_ = base::stringToDataType(data_type_str);
  }

  // 解析形状
  if (json.HasMember("shape_")) {
    const rapidjson::Value &shape_array = json["shape_"];
    if (shape_array.IsArray()) {
      shape_.clear();
      for (rapidjson::SizeType i = 0; i < shape_array.Size(); i++) {
        shape_.push_back(shape_array[i].GetInt());
      }
    }
  }

  return base::kStatusCodeOk;
}

ModelDesc::ModelDesc() {}
ModelDesc::~ModelDesc() {
  for (auto iter : weights_) {
    if (iter.second != nullptr) {
      delete iter.second;
    }
  }
}

/**
 * @brief
 *
 * @param oss
 * @return base::Status
 */
base::Status ModelDesc::dump(std::ostream &stream) {
  return serializeStructureToJson(stream);
}

/**
 * @brief 序列化模型结构为文本
 *
 * @param rapidjson::Value &json
 * @return base::Status
 * name_: 模型名称
 * metadata_: [{key, value}, {key, value}, ...]
 * inputs_: [ValueDesc0, ValueDesc1, ...]
 * outputs_: [ValueDesc0, ValueDesc1, ...]
 * op_descs_: [OpDesc0, OpDesc1, ...]
 * weights_: [name0, name1, ...]
 * values_: [ValueDesc0, ValueDesc1, ...]
 */
base::Status ModelDesc::serializeStructureToJson(
    rapidjson::Value &json,
    rapidjson::Document::AllocatorType &allocator) const {
  // 序列化模型名称
  json.AddMember("name_", rapidjson::Value(name_.c_str(), allocator),
                 allocator);

  // 序列化元数据
  if (!metadata_.empty()) {
    rapidjson::Value metadata_array(rapidjson::kArrayType);
    for (const auto &metadata : metadata_) {
      rapidjson::Value metadata_obj(rapidjson::kObjectType);
      metadata_obj.AddMember(
          rapidjson::Value(metadata.first.c_str(), allocator),
          rapidjson::Value(metadata.second.c_str(), allocator), allocator);
      metadata_array.PushBack(metadata_obj, allocator);
    }
    json.AddMember("metadata_", metadata_array, allocator);
  }

  // 序列化输入
  if (!inputs_.empty()) {
    rapidjson::Value inputs_array(rapidjson::kArrayType);
    for (const auto &input : inputs_) {
      rapidjson::Value input_json(rapidjson::kObjectType);
      input->serialize(input_json, allocator);
      inputs_array.PushBack(input_json, allocator);
    }
    json.AddMember("inputs_", inputs_array, allocator);
  }

  // 序列化输出
  if (!outputs_.empty()) {
    rapidjson::Value outputs_array(rapidjson::kArrayType);
    for (const auto &output : outputs_) {
      rapidjson::Value output_json(rapidjson::kObjectType);
      output->serialize(output_json, allocator);
      outputs_array.PushBack(output_json, allocator);
    }
    json.AddMember("outputs_", outputs_array, allocator);
  }

  // 序列化算子描述
  if (!op_descs_.empty()) {
    rapidjson::Value op_descs_array(rapidjson::kArrayType);
    for (const auto &op_desc : op_descs_) {
      rapidjson::Value op_desc_json(rapidjson::kObjectType);
      op_desc->serialize(op_desc_json, allocator);
      op_descs_array.PushBack(op_desc_json, allocator);
    }
    json.AddMember("op_descs_", op_descs_array, allocator);
  }

  // 序列化权重名称
  if (!weights_.empty()) {
    rapidjson::Value weights_array(rapidjson::kArrayType);
    for (const auto &weight : weights_) {
      weights_array.PushBack(rapidjson::Value(weight.first.c_str(), allocator),
                             allocator);
    }
    json.AddMember("weights_", weights_array, allocator);
  }

  // 序列化中间值
  if (!values_.empty()) {
    rapidjson::Value values_array(rapidjson::kArrayType);
    for (const auto &value : values_) {
      rapidjson::Value value_json(rapidjson::kObjectType);
      value->serialize(value_json, allocator);
      values_array.PushBack(value_json, allocator);
    }
    json.AddMember("values_", values_array, allocator);
  }

  return base::kStatusCodeOk;
}

base::Status ModelDesc::serializeStructureToJson(std::ostream &stream) const {
  rapidjson::Document doc;
  rapidjson::Value json(rapidjson::kObjectType);

  // 调用序列化函数
  base::Status status =
      this->serializeStructureToJson(json, doc.GetAllocator());
  if (status != base::kStatusCodeOk) {
    NNDEPLOY_LOGE("serializeStructureToJson failed with status: %d\n", status);
    return status;
  }

  // 检查文档是否为空
  // if (json.ObjectEmpty()) {
  //   NNDEPLOY_LOGE("Serialized JSON object is empty\n");
  //   return base::kStatusCodeErrorInvalidValue;
  // }

  // 序列化为字符串
  rapidjson::StringBuffer buffer;
  rapidjson::PrettyWriter<rapidjson::StringBuffer> writer(buffer);
  if (!json.Accept(writer)) {
    NNDEPLOY_LOGE("Failed to write JSON to buffer\n");
    return base::kStatusCodeErrorInvalidValue;
  }

  // 输出到流
  stream << buffer.GetString();
  // if (stream.fail()) {
  //   NNDEPLOY_LOGE("Failed to write JSON string to stream\n");
  //   return base::kStatusCodeErrorInvalidParam;
  // }

  return base::kStatusCodeOk;
}

base::Status ModelDesc::serializeStructureToJson(
    const std::string &path) const {
  std::ofstream ofs(path);
  if (!ofs.is_open()) {
    NNDEPLOY_LOGE("open file %s failed\n", path.c_str());
    return base::kStatusCodeErrorInvalidParam;
  }
  base::Status status = this->serializeStructureToJson(ofs);
  if (status != base::kStatusCodeOk) {
    NNDEPLOY_LOGE("serialize to json failed\n");
    return status;
  }
  ofs.close();
  return status;
}

/**
 * @brief
 * 从文本反序列化为模型结构，不包含weights和values，当input有值时，覆盖inputs_
 *
 * @param rapidjson::Value &json
 * @return base::Status
 * name_: 模型名称
 * metadata_: [{key, value}, {key, value}, ...]
 * inputs_: [ValueDesc0, ValueDesc1, ...]
 * outputs_: [ValueDesc0, ValueDesc1, ...]
 * op_descs_: [OpDesc0, OpDesc1, ...]
 * weights_: [name0, name1, ...](不实现)
 * values_: [ValueDesc0, ValueDesc1, ...](不实现)
 */
base::Status ModelDesc::deserializeStructureFromJson(
    rapidjson::Value &json, const std::vector<ValueDesc> &input) {
  // 反序列化模型名称
  if (json.HasMember("name_") && json["name_"].IsString()) {
    name_ = json["name_"].GetString();
  }

  // 反序列化元数据
  if (json.HasMember("metadata_") && json["metadata_"].IsArray()) {
    const rapidjson::Value &metadata_array = json["metadata_"];
    for (rapidjson::SizeType i = 0; i < metadata_array.Size(); i++) {
      if (metadata_array[i].IsObject()) {
        for (auto it = metadata_array[i].MemberBegin();
             it != metadata_array[i].MemberEnd(); ++it) {
          metadata_.insert({it->name.GetString(), it->value.GetString()});
        }
      }
    }
  }

  // 反序列化输入
  inputs_.clear();
  if (json.HasMember("inputs_") && json["inputs_"].IsArray()) {
    const rapidjson::Value &inputs_array = json["inputs_"];
    for (rapidjson::SizeType i = 0; i < inputs_array.Size(); i++) {
      auto value_desc = std::make_shared<ValueDesc>();
      value_desc->deserialize(const_cast<rapidjson::Value &>(inputs_array[i]));
      inputs_.push_back(value_desc);
    }
  }
  if (!input.empty()) {
    inputs_.clear();
    for (const auto &in : input) {
      for (auto &existing_input : inputs_) {
        if (existing_input->name_ == in.name_) {
          existing_input->name_ = in.name_;
          existing_input->data_type_ = in.data_type_;
          existing_input->shape_ = in.shape_;
          break;
        }
      }
    }
  }

  // 反序列化输出
  outputs_.clear();
  if (json.HasMember("outputs_") && json["outputs_"].IsArray()) {
    const rapidjson::Value &outputs_array = json["outputs_"];
    for (rapidjson::SizeType i = 0; i < outputs_array.Size(); i++) {
      auto value_desc = std::make_shared<ValueDesc>();
      value_desc->deserialize(const_cast<rapidjson::Value &>(outputs_array[i]));
      outputs_.push_back(value_desc);
    }
  }

  // 反序列化算子描述
  op_descs_.clear();
  if (json.HasMember("op_descs_") && json["op_descs_"].IsArray()) {
    const rapidjson::Value &op_descs_array = json["op_descs_"];
    for (rapidjson::SizeType i = 0; i < op_descs_array.Size(); i++) {
      auto op_desc = std::make_shared<OpDesc>();
      op_desc->deserialize(const_cast<rapidjson::Value &>(op_descs_array[i]));
      op_descs_.push_back(op_desc);
    }
  }

  return base::kStatusCodeOk;
}

base::Status ModelDesc::deserializeStructureFromJson(
    std::istream &stream, const std::vector<ValueDesc> &input) {
  std::string json_str;
  std::string line;
  while (std::getline(stream, line)) {
    json_str += line;
  }
  rapidjson::Document document;
  if (document.Parse(json_str.c_str()).HasParseError()) {
    NNDEPLOY_LOGE("parse json string failed\n");
    return base::kStatusCodeErrorInvalidParam;
  }
  rapidjson::Value &json = document;
  return this->deserializeStructureFromJson(json, input);
}
base::Status ModelDesc::deserializeStructureFromJson(
    const std::string &path, const std::vector<ValueDesc> &input) {
  std::ifstream ifs(path);
  if (!ifs.is_open()) {
    NNDEPLOY_LOGE("open file %s failed\n", path.c_str());
    return base::kStatusCodeErrorInvalidParam;
  }
  base::Status status = this->deserializeStructureFromJson(ifs, input);
  if (status != base::kStatusCodeOk) {
    NNDEPLOY_LOGE("deserialize from file %s failed\n", path.c_str());
    return status;
  }
  ifs.close();
  return status;
}

// 序列化模型权重为二进制文件
base::Status ModelDesc::serializeWeightsToSafetensorsImpl(
    safetensors::safetensors_t &st, bool serialize_buffer) const {
  for (auto &weight : weights_) {
    base::Status status =
        weight.second->serializeToSafetensors(st, serialize_buffer);
    NNDEPLOY_RETURN_VALUE_ON_NEQ(
        status, base::kStatusCodeOk, status,
        "weight.second->serializeToSafetensors(safetensors::safetensors_t "
        "&st, bool serialize_buffer)failed!\n");
  }
  return base::kStatusCodeOk;
}
/**
 * @brief 序列化模型权重为safetensors
 *
 * @param std::shared_ptr<safetensors::safetensors_t> &st_ptr(外部已经分配好了)
 * @return base::Status
 */
base::Status ModelDesc::serializeWeightsToSafetensors(
    std::shared_ptr<safetensors::safetensors_t> &serialize_st_ptr) const {
  base::Status status = base::kStatusCodeOk;

  if (metadata_.find("format") == metadata_.end()) {
    serialize_st_ptr->metadata.insert("format", "pt");
  } else {
    serialize_st_ptr->metadata.insert("format", metadata_.at("format"));
  }

  // 1. first record the tensor desc
  status = serializeWeightsToSafetensorsImpl(*serialize_st_ptr, false);
  NNDEPLOY_RETURN_VALUE_ON_NEQ(
      status, base::kStatusCodeOk, status,
      "model->serializeWeightsToSafetensorsImpl save param failed!\n");
  // 2. cacurate the storage size and insert the offsets, then we run second
  // time for copy or mapping data
  size_t total_storage_size = 0;
  safetensors::tensor_t t_t;
  for (int i = 0; i < serialize_st_ptr->tensors.size(); ++i) {
    serialize_st_ptr->tensors.at(i, &t_t);
    size_t tensor_size = safetensors::get_shape_size(t_t) *
                         safetensors::get_dtype_bytes(t_t.dtype);
    t_t.data_offsets = std::array<size_t, 2>{total_storage_size,
                                             total_storage_size + tensor_size};
    total_storage_size += tensor_size;
  }
  serialize_st_ptr->storage.resize(total_storage_size);

  // 3. real store buffer
  status = serializeWeightsToSafetensorsImpl(*serialize_st_ptr, true);
  NNDEPLOY_RETURN_VALUE_ON_NEQ(
      status, base::kStatusCodeOk, status,
      "model->serializeWeightsToSafetensorsImpl save buffer failed!\n");

  // 4. store metadata
  serialize_st_ptr->mmaped = false;

  return base::kStatusCodeOk;
}
base::Status ModelDesc::serializeWeightsToSafetensors(
    const std::string &weight_file_path) const {
  // 检查weight_file_path，确保使用'.safetensors'作为权重文件的后缀
  if (!weight_file_path.empty()) {
    std::string path = weight_file_path;
    const std::string extension = ".safetensors";
    size_t pos = weight_file_path.find_last_of('.');
    if (pos == std::string::npos || weight_file_path.substr(pos) != extension) {
      path = weight_file_path + extension;
    }
    std::shared_ptr<safetensors::safetensors_t> st_ptr(
        new safetensors::safetensors_t());

    base::Status status = this->serializeWeightsToSafetensors(st_ptr);
    if (status != base::kStatusCodeOk) {
      NNDEPLOY_LOGE("model_desc_->serializeWeightsToBinary failed!\n");
      return status;
    }

    std::string warn, err;
    bool ret = safetensors::save_to_file((*st_ptr), path, &warn, &err);
    if (warn.size()) {
      NNDEPLOY_LOGI("WARN: %s\n", warn.c_str());
    }
    if (!ret) {
      NNDEPLOY_LOGE("Failed to load: %s\nERR: %s", path.c_str(), err.c_str());
      return base::kStatusCodeErrorIO;
    }
  }
  return base::kStatusCodeOk;
}

// 从safetensors导入成模型文件
base::Status ModelDesc::deserializeWeightsFromSafetensors(
    std::shared_ptr<safetensors::safetensors_t> &st_ptr) {
  base::Status status = base::kStatusCodeOk;
  // 释放之前权重
  for (auto &weight : weights_) {
    if (weight.second != nullptr) {
      delete weight.second;
    }
  }
  weights_.clear();
  // 导入权重
  size_t tensor_size = st_ptr->tensors.size();
  const std::vector<std::string> &keys = st_ptr->tensors.keys();
  for (size_t i = 0; i < tensor_size; ++i) {
    std::string tensor_name = keys[i];
    weights_[tensor_name] = new device::Tensor(tensor_name);
    if (weights_[tensor_name] == nullptr) {
      NNDEPLOY_LOGE("new device::Tensor failed\n");
      return base::kStatusCodeErrorOutOfMemory;
    }
    status = weights_[tensor_name]->serializeFromSafetensors(*st_ptr);
    NNDEPLOY_RETURN_VALUE_ON_NEQ(
        status, base::kStatusCodeOk, status,
        "The model file and the weight file do not match. !\n");
  }
  return status;
}

}  // namespace ir
}  // namespace nndeploy
