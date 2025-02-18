

#include "nndeploy/ir/onnx/onnx_interpret.h"

#include <google/protobuf/message.h>
#include <google/protobuf/text_format.h>

#include "google/protobuf/io/coded_stream.h"
#include "google/protobuf/io/zero_copy_stream_impl.h"
#include "nndeploy/ir/ir.h"

namespace nndeploy {
namespace ir {

TypeInterpretRegister<TypeInterpretCreator<OnnxInterpret>>
    g_onnx_interpret_register(base::kModelTypeOnnx);

OnnxInterpret::OnnxInterpret(ModelDesc* model_desc, bool is_external)
    : Interpret(model_desc, is_external) {}
OnnxInterpret::~OnnxInterpret() {}

// convert
base::DataType OnnxInterpret::convertToDataType(
    const onnx::TensorProto_DataType& src) {
  base::DataType dst;
  switch (src) {
    case onnx::TensorProto_DataType::TensorProto_DataType_FLOAT:
      dst = base::dataTypeOf<float>();
      break;
    case onnx::TensorProto_DataType::TensorProto_DataType_UINT8:
      dst = base::dataTypeOf<uint8_t>();
      break;
    case onnx::TensorProto_DataType::TensorProto_DataType_INT8:
      dst = base::dataTypeOf<int8_t>();
      break;
    case onnx::TensorProto_DataType::TensorProto_DataType_UINT16:
      dst = base::dataTypeOf<uint16_t>();
      break;
    case onnx::TensorProto_DataType::TensorProto_DataType_INT16:
      dst = base::dataTypeOf<int16_t>();
      break;
    case onnx::TensorProto_DataType::TensorProto_DataType_INT32:
      dst = base::dataTypeOf<int32_t>();
      break;
    case onnx::TensorProto_DataType::TensorProto_DataType_INT64:
      dst = base::dataTypeOf<int64_t>();
      break;
    case onnx::TensorProto_DataType::TensorProto_DataType_BOOL:
      dst = base::dataTypeOf<uint8_t>();
      break;
    case onnx::TensorProto_DataType::TensorProto_DataType_FLOAT16:
      dst.code_ = base::kDataTypeCodeFp;
      dst.bits_ = 16;
      dst.lanes_ = 1;
      break;
    case onnx::TensorProto_DataType::TensorProto_DataType_DOUBLE:
      dst = base::dataTypeOf<double>();
      break;
    case onnx::TensorProto_DataType::TensorProto_DataType_UINT32:
      dst = base::dataTypeOf<uint32_t>();
      break;
    case onnx::TensorProto_DataType::TensorProto_DataType_UINT64:
      dst = base::dataTypeOf<uint64_t>();
      break;
    case onnx::TensorProto_DataType::TensorProto_DataType_BFLOAT16:
      dst.code_ = base::kDataTypeCodeBFp;
      dst.bits_ = 16;
      dst.lanes_ = 1;
      break;
    default:  // 要不要扩充呢？
      NNDEPLOY_LOGE("Not support onnx TypeProto type: %d", src);
      break;
  }
  return dst;
}
base::IntVector OnnxInterpret::convertToShape(
    const onnx::TensorShapeProto& src) {
  base::IntVector dst;
  int dim_size = src.dim_size();
  // TODO 是否要给一个一维的默认值呢
  if (dim_size == 0) {
    dst.push_back(1);
  } else {
    for (int i = 0; i < dim_size; ++i) {
      dst.push_back(src.dim(i).dim_value());
    }
  }
  return dst;
}
base::DataFormat OnnxInterpret::convertToDataFormat(
    const onnx::TensorShapeProto& src, bool is_weight) {
  base::DataFormat dst;
  int dim_size = src.dim_size();
  if (dim_size <= 1) {
    dst = base::kDataFormatNC;
  } else if (dim_size == 2) {
    dst = base::kDataFormatNC;
  } else if (dim_size == 3) {
    dst = base::kDataFormatNCL;
  } else if (dim_size == 4) {
    if (is_weight) {
      dst = base::kDataFormatOIHW;
    } else {
      dst = base::kDataFormatNCHW;
    }
  } else if (dim_size == 5) {
    dst = base::kDataFormatNCDHW;
  } else {
    NNDEPLOY_LOGE("Not support onnx TensorShapeProto dim size: %d", dim_size);
  }
  return dst;
}

base::IntVector OnnxInterpret::convertToShape(
    const google::protobuf::RepeatedField<::int64_t>& src) {
  base::IntVector dst;
  int dim_size = src.size();
  // TODO 是否要给一个一维的默认值呢
  if (dim_size == 0) {
    dst.push_back(1);
  } else {
    for (int i = 0; i < dim_size; ++i) {
      dst.push_back(src[i]);
    }
  }
  return dst;
}
base::DataFormat OnnxInterpret::convertToDataFormat(
    const google::protobuf::RepeatedField<::int64_t>& src, bool is_weight) {
  base::DataFormat dst;
  int dim_size = src.size();
  if (dim_size <= 1) {
    dst = base::kDataFormatNC;
  } else if (dim_size == 2) {
    dst = base::kDataFormatNC;
  } else if (dim_size == 3) {
    dst = base::kDataFormatNCL;
  } else if (dim_size == 4) {
    if (is_weight) {
      dst = base::kDataFormatOIHW;
    } else {
      dst = base::kDataFormatNCHW;
    }
  } else if (dim_size == 5) {
    dst = base::kDataFormatNCDHW;
  } else {
    NNDEPLOY_LOGE("Not support onnx TensorShapeProto dim size: %d", dim_size);
  }
  return dst;
}
std::shared_ptr<OpDesc> OnnxInterpret::convertToOpDesc(
    const onnx::NodeProto& src) {
  auto convert = createOnnxOpConvert(src.op_type());
  if (convert == nullptr) {
    NNDEPLOY_LOGE("Not support onnx op type: %s\n", src.op_type().c_str());
    return nullptr;
  }
  std::shared_ptr<OpDesc> dst = convert->convert(src);
  return dst;
}
device::Tensor* OnnxInterpret::convertToTensor(const onnx::TensorProto& src) {
  std::string name = src.name();
  // NNDEPLOY_LOGI("name = %s\n", name.c_str());
  device::TensorDesc desc;
  onnx::TensorProto_DataType onnx_data_type =
      (onnx::TensorProto_DataType)src.data_type();
  desc.data_type_ = convertToDataType(onnx_data_type);
  desc.data_format_ = convertToDataFormat(src.dims(), true);
  desc.shape_ = convertToShape(src.dims());
  void* data_ptr = getDataFromTensor(src);
  // debug
  // NNDEPLOY_LOGI("data_ptr = %p\n", data_ptr);
  // desc.print();
  device::Device* device = device::getDefaultHostDevice();
  device::Tensor* tensor = new device::Tensor(device, desc, data_ptr, name);
  return tensor;
}
device::Tensor* OnnxInterpret::convertToTensor(const onnx::TensorProto& src,
                                               const std::string& name) {
  device::TensorDesc desc;
  onnx::TensorProto_DataType onnx_data_type =
      (onnx::TensorProto_DataType)src.data_type();
  desc.data_type_ = convertToDataType(onnx_data_type);
  desc.data_format_ = convertToDataFormat(src.dims(), true);
  desc.shape_ = convertToShape(src.dims());
  void* data_ptr = getDataFromTensor(src);
  // debug
  // NNDEPLOY_LOGI("data_ptr = %p\n", data_ptr);
  // desc.print();
  device::Device* device = device::getDefaultHostDevice();
  device::Tensor* tensor = new device::Tensor(device, desc, data_ptr, name);
  return tensor;
}
std::shared_ptr<ValueDesc> OnnxInterpret::convertToValueDesc(
    const onnx::ValueInfoProto& src) {
  std::string name = src.name();
  // 判断是否存在
  onnx::TensorProto_DataType onnx_data_type =
      (onnx::TensorProto_DataType)src.type().tensor_type().elem_type();
  base::DataType data_type = convertToDataType(onnx_data_type);
  // 判断是否存在
  base::IntVector shape = convertToShape(src.type().tensor_type().shape());
  return std::make_shared<ValueDesc>(name, data_type, shape);
}

onnx::AttributeProto_AttributeType OnnxInterpret::getAttributeType(
    const char* type_name) {
  // 需要补充类型吗？
  if (type_name == typeid(int64_t).name()) {
    return onnx::AttributeProto_AttributeType_INT;
  } else if (type_name == typeid(int64_t[]).name()) {
    return onnx::AttributeProto_AttributeType_INTS;
  } else if (type_name == typeid(float).name()) {
    return onnx::AttributeProto_AttributeType_FLOAT;
  } else if (type_name == typeid(float[]).name()) {
    return onnx::AttributeProto_AttributeType_FLOATS;
  } else if (type_name == typeid(std::string).name()) {
    return onnx::AttributeProto_AttributeType_STRING;
  } else if (type_name == typeid(std::string[]).name()) {
    return onnx::AttributeProto_AttributeType_STRINGS;
  } else if (type_name == typeid(onnx::TensorProto).name()) {
    return onnx::AttributeProto_AttributeType_TENSOR;
  } else if (type_name == typeid(onnx::TensorProto[]).name()) {
    return onnx::AttributeProto_AttributeType_TENSORS;
  } else if (type_name == typeid(onnx::GraphProto).name()) {
    return onnx::AttributeProto_AttributeType_GRAPH;
  } else if (type_name == typeid(onnx::GraphProto[]).name()) {
    return onnx::AttributeProto_AttributeType_GRAPHS;
  } else {
    return onnx::AttributeProto_AttributeType_UNDEFINED;
  }
}
int32_t OnnxInterpret::getAttributeInt(const onnx::NodeProto& node,
                                       const std::string& name,
                                       int32_t default_value) {
  for (const auto& iter : node.attribute()) {
    if (iter.name() != name) {
      continue;
    }
    assert(iter.type() == onnx::AttributeProto_AttributeType_INT);
    return iter.i();
  }
  return default_value;
}
std::vector<int32_t> OnnxInterpret::getAttributeIntVector(
    const onnx::NodeProto& node, const std::string& name) {
  std::vector<int32_t> attributes;
  for (const auto& iter : node.attribute()) {
    if (iter.name() != name) {
      continue;
    }
    assert(iter.type() == onnx::AttributeProto_AttributeType_INTS);
    for (const auto& value : iter.ints()) {
      attributes.push_back(value);
    }
  }
  return attributes;
}
std::vector<int64_t> OnnxInterpret::getAttributeInt64Vector(
    const onnx::NodeProto& node, const std::string& name) {
  std::vector<int64_t> attributes;
  for (const auto& iter : node.attribute()) {
    if (iter.name() != name) {
      continue;
    }
    assert(iter.type() == onnx::AttributeProto_AttributeType_INTS);
    for (const auto& value : iter.ints()) {
      attributes.push_back(value);
    }
  }
  return attributes;
}
float OnnxInterpret::getAttributeFloat(const onnx::NodeProto& node,
                                       const std::string& name,
                                       float default_value) {
  for (const auto& iter : node.attribute()) {
    if (iter.name() != name) {
      continue;
    }
    assert(iter.type() == onnx::AttributeProto_AttributeType_FLOAT);
    return iter.f();
  }
  return default_value;
}
std::string OnnxInterpret::getAttributeString(const onnx::NodeProto& node,
                                              const std::string& name,
                                              std::string def) {
  for (const auto& iter : node.attribute()) {
    if (iter.name() == name) {
      assert(iter.type() == onnx::AttributeProto_AttributeType_STRING);
      return iter.s();
    }
  }
  return def;
}
std::vector<std::string> OnnxInterpret::getAttributeStringVector(
    const onnx::NodeProto& node, const std::string& name) {
  std::vector<std::string> attributes;
  for (const auto& iter : node.attribute()) {
    if (iter.name() != name) {
      continue;
    }
    assert(iter.type() == onnx::AttributeProto_AttributeType_STRINGS);
    for (const auto& value : iter.strings()) {
      attributes.push_back(value);
    }
  }
  return attributes;
}

std::vector<std::string> OnnxInterpret::splitString(std::string& s,
                                                    const std::string& c) {
  std::vector<std::string> res;
  std::string::size_type pos1, pos2;
  pos2 = s.find(c);
  pos1 = 0;
  while (std::string::npos != pos2) {
    res.push_back(s.substr(pos1, pos2 - pos1));

    pos1 = pos2 + c.size();
    pos2 = s.find(c, pos1);
  }
  if (pos1 != s.length()) {
    res.push_back(s.substr(pos1));
  }
  return res;
}

std::vector<uint8_t> OnnxInterpret::getAttributeUInt8Vector(
    const onnx::NodeProto& node, const std::string& name) {
  std::vector<uint8_t> attribute;
  for (const auto& iter : node.attribute()) {
    if (iter.name() == name) {
      assert(iter.type() == onnx::AttributeProto_AttributeType_STRING);
      const auto& raw_data = iter.s();
      int size = raw_data.size();
      for (int i = 0; i < size; ++i) {
        attribute.push_back(*((uint8_t*)raw_data.data() + i));
      }
    }
  }
  return attribute;
}

std::vector<int8_t> OnnxInterpret::asymmetric2Symmetric(
    std::vector<uint8_t>& raw_value, uint8_t zero_point) {
  std::vector<int8_t> res;
  for (const auto& value : raw_value) {
    res.push_back(value - zero_point);
  }
  return res;
}

onnx::TensorProto OnnxInterpret::getAttributeTensor(const onnx::NodeProto& node,
                                                    const char* key) {
  for (int i = 0; i < node.attribute_size(); i++) {
    const onnx::AttributeProto& attr = node.attribute(i);
    if (attr.name() == key) {
      return attr.t();
    }
  }

  return onnx::TensorProto();
}

int OnnxInterpret::getTensorProtoDataSize(const onnx::TensorProto& tp) {
  int size = 0;
  if (tp.has_raw_data()) {
    const std::string& raw_data = tp.raw_data();
    switch (tp.data_type()) {
      case onnx::TensorProto_DataType_FLOAT: {
        size = int(raw_data.size() / sizeof(float));
        break;
      }
      case onnx::TensorProto_DataType_UINT8: {
        size = int(raw_data.size() / sizeof(uint8_t));
        break;
      }
      case onnx::TensorProto_DataType_INT8: {
        size = int(raw_data.size() / sizeof(int8_t));
        break;
      }
      case onnx::TensorProto_DataType_UINT16: {
        size = int(raw_data.size() / sizeof(uint16_t));
        break;
      }
      case onnx::TensorProto_DataType_INT16: {
        size = int(raw_data.size() / sizeof(int16_t));
        break;
      }
      case onnx::TensorProto_DataType_INT32: {
        size = int(raw_data.size() / sizeof(int32_t));
        break;
      }
      case onnx::TensorProto_DataType_INT64: {
        size = int(raw_data.size() / sizeof(int64_t));
        break;
      }
      case onnx::TensorProto_DataType_BOOL: {
        size = int(raw_data.size() / sizeof(bool));
        break;
      }
      case onnx::TensorProto_DataType_FLOAT16: {
        size = int(raw_data.size() / (sizeof(float) / 2));
        break;
      }
      case onnx::TensorProto_DataType_DOUBLE: {
        size = int(raw_data.size() / sizeof(double));
        break;
      }
      case onnx::TensorProto_DataType_UINT32: {
        size = int(raw_data.size() / sizeof(uint32_t));
        break;
      }
      case onnx::TensorProto_DataType_UINT64: {
        size = int(raw_data.size() / sizeof(uint64_t));
        break;
      }
      default: {
        NNDEPLOY_LOGE(
            "Onnx Converter: do not support tensor proto data type\n");
        size = -1;
      }
    }
  } else {
    switch (tp.data_type()) {
      case onnx::TensorProto_DataType_FLOAT: {
        size = tp.float_data_size();
        break;
      }
      case onnx::TensorProto_DataType_INT32: {
        size = tp.int32_data_size();
        break;
      }
      case onnx::TensorProto_DataType_INT64: {
        size = tp.int64_data_size();
        break;
      }
      case onnx::TensorProto_DataType_DOUBLE: {
        size = tp.double_data_size();
        break;
      }
      default: {
        NNDEPLOY_LOGE(
            "Onnx Converter: do not support tensor proto data type\n");
        size = -1;
      }
    }
  }
  return size;
}

void* OnnxInterpret::getDataFromTensor(const onnx::TensorProto& tensor) {
  void* data_ptr = nullptr;
  onnx::TensorProto_DataType data_type =
      (onnx::TensorProto_DataType)tensor.data_type();
  if (tensor.has_raw_data()) {
    data_ptr = (void*)tensor.raw_data().data();
  } else if (data_type == onnx::TensorProto_DataType_FLOAT) {
    data_ptr = (void*)tensor.float_data().data();
  } else if (data_type == onnx::TensorProto_DataType_INT32) {
    data_ptr = (void*)tensor.int32_data().data();
  } else if (data_type == onnx::TensorProto_DataType_INT64) {
    data_ptr = (void*)tensor.int64_data().data();
  } else if (data_type == onnx::TensorProto_DataType_DOUBLE) {
    data_ptr = (void*)(tensor.double_data().data());
  } else {
    NNDEPLOY_LOGE("Tensor(%s) do not have valid data\n", tensor.name().c_str());
  }
  return data_ptr;
}

const onnx::TensorProto* OnnxInterpret::getTensorFromConstantNode(
    const onnx::NodeProto& constant_node) {
  for (int i = 0; i < constant_node.attribute_size(); ++i) {
    const auto& attribute_proto = constant_node.attribute(i);
    const auto& attribute_name = attribute_proto.name();
    if (attribute_name == "value") {
      return &attribute_proto.t();
    }
  }
  return nullptr;
}

base::Status OnnxInterpret::interpret(
    const std::vector<std::string>& model_value,
    const std::vector<ValueDesc>& input) {
  base::Status status = base::kStatusCodeOk;

  // 读模型文件
  std::ifstream input_stream(model_value[0],
                             std::ifstream::in | std::ifstream::binary);
  if (!input_stream.is_open()) {
    NNDEPLOY_LOGE("model_value[%s] is error.\n", model_value[0].c_str());
    return base::kStatusCodeErrorInvalidParam;
  }

  google::protobuf::io::IstreamInputStream proto_input_stream(&input_stream);
  google::protobuf::io::CodedInputStream coded_input_stream(
      &proto_input_stream);
#if GOOGLE_PROTOBUF_VERSION >= 3002000
  coded_input_stream.SetTotalBytesLimit(INT_MAX);
#else
  coded_input_stream.SetTotalBytesLimit(INT_MAX, INT_MAX / 2);
#endif
  this->onnx_model_ = std::unique_ptr<onnx::ModelProto>(new onnx::ModelProto());
  bool success = this->onnx_model_->ParseFromCodedStream(&coded_input_stream);
  if (!success) {
    NNDEPLOY_LOGE(
        "this->onnx_model_->ParseFromCodedStream(&coded_input_stream) "
        "failed.\n");
    return base::kStatusCodeErrorInvalidParam;
  }
  input_stream.close();

  // 检查并转换ONNX模型版本
  if (this->onnx_model_->ir_version() != target_version_) {
    NNDEPLOY_LOGI("current version: %lld, target_version_ %d.\n",
                  this->onnx_model_->ir_version(), target_version_);
    try {
      onnx::ModelProto converted_model =
          onnx::version_conversion::ConvertVersion(*(this->onnx_model_),
                                                   target_version_);
      // Store the ONNX model
      std::ofstream output_stream("converted_model.onnx",
                                  std::ofstream::out | std::ofstream::binary);
      if (!output_stream.is_open()) {
        NNDEPLOY_LOGE("Failed to open the output file.\n");
        return base::kStatusCodeErrorInvalidParam;
      }
      if (!converted_model.SerializeToOstream(&output_stream)) {
        NNDEPLOY_LOGE("Failed to serialize the ONNX model.\n");
        return base::kStatusCodeErrorInvalidParam;
      }
      output_stream.close();
      *(this->onnx_model_) = std::move(converted_model);
      NNDEPLOY_LOGI("Model version successfully converted to %d.\n",
                    target_version_);
    } catch (const std::exception& e) {
      NNDEPLOY_LOGE("Error occurred during version conversion: %s.\n",
                    e.what());
      return base::kStatusCodeErrorInvalidParam;
    }
  }

  // 解析图
  const auto& onnx_graph = onnx_model_->graph();

  // # 解析名字
  model_desc_->name_ = onnx_model_->graph().name();

  // # 解析输入
  const int input_size = onnx_graph.input_size();
  NNDEPLOY_LOGE("input_size = %d\n", input_size);
  for (int i = 0; i < input_size; ++i) {
    const auto& input = onnx_graph.input(i);

    std::shared_ptr<ValueDesc> value_desc = convertToValueDesc(input);

    model_desc_->inputs_.push_back(value_desc);
  }

  // # 解析输出
  const int output_size = onnx_graph.output_size();
  NNDEPLOY_LOGE("output_size = %d\n", output_size);
  for (int i = 0; i < output_size; ++i) {
    const auto& output = onnx_graph.output(i);
    std::shared_ptr<ValueDesc> value_desc = convertToValueDesc(output);
    model_desc_->outputs_.push_back(value_desc);
  }

  // # 解析权重
  const int initializer_size = onnx_graph.initializer_size();
  NNDEPLOY_LOGE("initializer_size = %d\n", initializer_size);
  for (int i = 0; i < initializer_size; ++i) {
    const auto& initializer = onnx_graph.initializer(i);
    std::string name = initializer.name();
    // NNDEPLOY_LOGI("initializer name = %s\n", name.c_str());
    // 浅拷贝权重数据
    device::Tensor* tensor = convertToTensor(initializer);
    model_desc_->weights_.insert(std::make_pair(name, tensor));
  }

  // # 节点个数
  const int node_size = onnx_graph.node_size();
  NNDEPLOY_LOGE("node_size = %d\n", node_size);
  // ## 解析const节点
  for (int i = 0; i < node_size; ++i) {
    const auto& onnx_node = onnx_graph.node(i);
    const std::string& node_op_type = onnx_node.op_type();
    if (node_op_type == "Constant") {
      const onnx::TensorProto* initializer =
          getTensorFromConstantNode(onnx_node);
      std::string name = onnx_node.output(0);  // 非常重要

      device::Tensor* tensor = convertToTensor(*initializer, name);
      model_desc_->weights_.insert(std::make_pair(name, tensor));
    }
  }
  // ## 解析节点
  for (int i = 0; i < node_size; ++i) {
    const auto& onnx_node = onnx_graph.node(i);
    const std::string& node_op_type = onnx_node.op_type();
    if (node_op_type == "Constant") {
      continue;
    }
    // NNDEPLOY_LOGE("node_op_type = %s\n", node_op_type.c_str());
    std::shared_ptr<OpDesc> op_desc = convertToOpDesc(onnx_node);
    if (op_desc == nullptr) {
      NNDEPLOY_LOGE("convertToOpDesc error: %s\n", node_op_type.c_str());
      return base::kStatusCodeErrorInvalidParam;
    }
    model_desc_->op_descs_.push_back(op_desc);
  }

  return status;
}

std::map<std::string, std::shared_ptr<OnnxOpConvertCreator>>&
getGlobalOnnxOpConvertCreatorMap() {
  static std::once_flag once;
  static std::shared_ptr<
      std::map<std::string, std::shared_ptr<OnnxOpConvertCreator>>>
      creators;
  std::call_once(once, []() {
    creators.reset(
        new std::map<std::string, std::shared_ptr<OnnxOpConvertCreator>>);
  });
  return *creators;
}

std::shared_ptr<OnnxOpConvert> createOnnxOpConvert(const std::string& type) {
  std::shared_ptr<OnnxOpConvert> temp = nullptr;
  auto& creater_map = getGlobalOnnxOpConvertCreatorMap();
  if (creater_map.count(type) > 0) {
    temp = creater_map[type]->createOnnxOpConvert(type);
  }
  return temp;
}

}  // namespace ir
}  // namespace nndeploy
