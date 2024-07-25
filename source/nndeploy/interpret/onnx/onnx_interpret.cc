

#include "nndeploy/interpret/onnx/onnx_interpret.h"

#include <google/protobuf/message.h>
#include <google/protobuf/text_format.h>

#include "google/protobuf/io/coded_stream.h"
#include "google/protobuf/io/zero_copy_stream_impl.h"
#include "nndeploy/op/ir.h"
#include "nndeploy/op/op.h"

namespace nndeploy {
namespace interpret {

OnnxInterpret::OnnxInterpret() : Interpret() {}
OnnxInterpret::~OnnxInterpret() {}

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
    default:
      NNDEPLOY_LOGE("Not support onnx TypeProto type: %d", src);
      assert(0);
      break;
  }
  return dst;
}
base::IntVector OnnxInterpret::convertToShape(
    const onnx::TensorShapeProto& src) {
  base::IntVector dst;
  int dim_size = src.dim_size();
  for (int i = 0; i < dim_size; ++i) {
    dst.push_back(src.dim(i).dim_value());
  }
  return dst;
}
base::DataFormat OnnxInterpret::convertToDataFormat(
    const onnx::TensorShapeProto& src, bool is_weight) {
  base::DataFormat dst;
  int dim_size = src.dim_size();
  if (dim_size == 1) {
    dst = base::kDataFormatN;
  } else if (dim_size == 2) {
    dst = base::kDataFormatNC;
  } else if (dim_size == 3) {
    dst = base::kDataFormatNCW;
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
    assert(0);
  }
  return dst;
}

base::IntVector OnnxInterpret::convertToShape(
    const google::protobuf::RepeatedField<::int64_t>& src) {
  base::IntVector dst;
  int dim_size = src.size();
  for (int i = 0; i < dim_size; ++i) {
    dst.push_back(src[i]);
  }
  return dst;
}
base::DataFormat OnnxInterpret::convertToDataFormat(
    const google::protobuf::RepeatedField<::int64_t>& src, bool is_weight) {
  base::DataFormat dst;
  int dim_size = src.size();
  if (dim_size == 1) {
    dst = base::kDataFormatN;
  } else if (dim_size == 2) {
    dst = base::kDataFormatNC;
  } else if (dim_size == 3) {
    dst = base::kDataFormatNCW;
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
    assert(0);
  }
  return dst;
}
std::shared_ptr<op::OpDesc> OnnxInterpret::convertToOpDesc(
    const onnx::NodeProto& src) {
  auto convert = createOnnxOpConvert(src.op_type());
  std::shared_ptr<op::OpDesc> dst = convert->convert(src);
  return dst;
}
device::Tensor* OnnxInterpret::convertToTensor(const onnx::TensorProto& src) {
  std::string name = src.name();
  device::TensorDesc desc;
  onnx::TensorProto_DataType onnx_data_type =
      (onnx::TensorProto_DataType)src.data_type();
  desc.data_type_ = convertToDataType(onnx_data_type);
  desc.data_format_ = convertToDataFormat(src.dims(), true);
  desc.shape_ = convertToShape(src.dims());
  void* data_ptr = (void*)src.raw_data().data();
  device::Device* device = device::getDefaultHostDevice();
  device::Tensor* tensor = new device::Tensor(device, desc, data_ptr, name);
  return tensor;
}
std::shared_ptr<op::ValueDesc> OnnxInterpret::convertToValueDesc(
    const onnx::ValueInfoProto& src) {
  std::string name = src.name();
  onnx::TensorProto_DataType onnx_data_type =
      (onnx::TensorProto_DataType)src.type().tensor_type().elem_type();
  base::DataType data_type = convertToDataType(onnx_data_type);
  base::IntVector shape = convertToShape(src.type().tensor_type().shape());
  return std::make_shared<op::ValueDesc>(name, data_type, shape);
}

base::Status OnnxInterpret::interpret(
    const std::vector<std::string>& model_value,
    const std::vector<op::ValueDesc>& input) {
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
  input_stream.close();

  // 解析图
  const auto& onnx_graph = onnx_model_->graph();
  // # 解析节点
  const int node_size = onnx_graph.node_size();
  for (int i = 0; i < node_size; ++i) {
    const auto& onnx_node = onnx_graph.node(i);
    std::shared_ptr<op::OpDesc> op_desc = convertToOpDesc(onnx_node);
    model_desc_->op_descs_.push_back(op_desc);
  }
  // # 解析权重
  const int initializer_size = onnx_graph.initializer_size();
  for (int i = 0; i < initializer_size; ++i) {
    const auto& initializer = onnx_graph.initializer(i);
    std::string name = initializer.name();
    // 浅拷贝权重数据
    device::Tensor* tensor = convertToTensor(initializer);
    model_desc_->weights_.insert(std::make_pair(name, tensor));
  }
  // # 解析输入
  const int input_size = onnx_graph.input_size();
  for (int i = 0; i < input_size; ++i) {
    const auto& input = onnx_graph.input(i);
    std::shared_ptr<op::ValueDesc> value_desc = convertToValueDesc(input);
    model_desc_->inputs_.push_back(value_desc);
  }
  // 解析输出
  const int output_size = onnx_graph.output_size();
  for (int i = 0; i < output_size; ++i) {
    const auto& output = onnx_graph.output(i);
    std::shared_ptr<op::ValueDesc> value_desc = convertToValueDesc(output);
    model_desc_->outputs_.push_back(value_desc);
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

}  // namespace interpret
}  // namespace nndeploy
